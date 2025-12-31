/// <reference lib="webworker" />

import { Chess, Move } from 'chess.js';
import * as tf from '@tensorflow/tfjs';

import { compileTinyZeroModel, createTinyZeroModel } from '../services/model';
import { runMcts } from '../services/mcts';
import { boardToTensor } from '../services/tensorUtils';
import { moveToIndex } from '../services/moveEncoding';
import {
  POLICY_OUTPUT_SIZE,
  TRAINING_MCTS_SIMULATIONS,
  TRAINING_DIRICHLET_ALPHA,
  TRAINING_DIRICHLET_EPSILON,
  MAX_REPLAY_BUFFER,
  BATCH_SIZE
} from '../constants';
import type {
  TrainingMetrics,
  MoveProbability,
  HeatmapSquare,
  MctsDifficulty,
  MctsConfig,
  TrainingSample,
  MoveLike
} from '../types';

const INITIAL_METRICS: TrainingMetrics = {
  epoch: 0,
  gamesPlayed: 0,
  policyLoss: 2.5,
  valueLoss: 1.0,
  winRate: 0.1,
  entropy: 4.5
};

const MCTS_DIFFICULTY: Record<MctsDifficulty, MctsConfig> = {
  easy: { simulations: 80, cPuct: 1.2, temperature: 1.1 },
  medium: { simulations: 200, cPuct: 1.4, temperature: 0.8 },
  hard: { simulations: 600, cPuct: 1.6, temperature: 0.4 }
};

const TRAINING_CONFIG: MctsConfig = {
  simulations: TRAINING_MCTS_SIMULATIONS,
  cPuct: 1.5,
  temperature: 1.0,
  dirichletAlpha: TRAINING_DIRICHLET_ALPHA,
  dirichletEpsilon: TRAINING_DIRICHLET_EPSILON
};

type WorkerStatus = 'loading' | 'ready' | 'error';

type InitMessage = {
  type: 'init';
  baseModelUrl: string;
};

type StartMessage = { type: 'start' };

type StopMessage = { type: 'stop' };

type ResetMessage = { type: 'reset' };

type SetDifficultyMessage = {
  type: 'setDifficulty';
  difficulty: MctsDifficulty;
};

type WorkerMessage =
  | InitMessage
  | StartMessage
  | StopMessage
  | ResetMessage
  | SetDifficultyMessage;

type StatusPayload = {
  type: 'status';
  status: WorkerStatus;
  error?: string;
};

type StatePayload = {
  type: 'state';
  fen: string;
  heatmap: HeatmapSquare[];
  topMoves: MoveProbability[];
  moveHistory: string[];
  currentMetrics: TrainingMetrics;
  metricsHistory: TrainingMetrics[];
  isMctsThinking: boolean;
};

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

let model: tf.LayersModel | null = null;
let modelStatus: WorkerStatus = 'loading';
let modelError: string | undefined;

let difficulty: MctsDifficulty = 'medium';
let isTraining = false;
let isMctsThinking = false;
let loopTimeout: number | null = null;

const game = new Chess();
const replayBuffer: TrainingSample[] = [];
let currentGameSamples: TrainingSample[] = [];
let metricsHistory: TrainingMetrics[] = [];
let currentMetrics: TrainingMetrics = { ...INITIAL_METRICS };
let heatmap: HeatmapSquare[] = [];
let topMoves: MoveProbability[] = [];
let moveHistory: string[] = [];

const postStatus = () => {
  const payload: StatusPayload = {
    type: 'status',
    status: modelStatus,
    error: modelError
  };
  ctx.postMessage(payload);
};

const postState = () => {
  const payload: StatePayload = {
    type: 'state',
    fen: game.fen(),
    heatmap,
    topMoves,
    moveHistory,
    currentMetrics,
    metricsHistory,
    isMctsThinking
  };
  ctx.postMessage(payload);
};

const sampleFromPolicy = (policy: number[]): number => {
  let threshold = Math.random();
  for (let i = 0; i < policy.length; i++) {
    threshold -= policy[i];
    if (threshold <= 0) return i;
  }
  return policy.length - 1;
};

const computeEntropy = (policy: number[]): number => {
  return policy.reduce((acc, p) => (p > 0 ? acc - p * Math.log(p) : acc), 0);
};

const serializeMove = (move: Move): MoveLike => ({
  from: move.from,
  to: move.to,
  promotion: move.promotion,
  san: move.san
});

const setThinking = (value: boolean) => {
  if (isMctsThinking === value) return;
  isMctsThinking = value;
  postState();
};

const runMctsWithIndicator = async (
  gameState: Chess,
  activeModel: tf.LayersModel,
  config: MctsConfig,
  addNoise = false
) => {
  setThinking(true);
  try {
    return await runMcts(gameState, activeModel, config, addNoise);
  } finally {
    setThinking(false);
  }
};

const resetGameState = () => {
  game.reset();
  heatmap = [];
  topMoves = [];
  moveHistory = [];
  postState();
};

const stepTraining = async () => {
  if (!model) return;
  if (game.isGameOver()) return;

  const turn = game.turn();
  let selectedMove: Move | null = null;

  if (turn === 'w') {
    const mcts = await runMctsWithIndicator(game, model, TRAINING_CONFIG, true);
    const policyVector = new Array(POLICY_OUTPUT_SIZE).fill(0);
    mcts.policy.forEach((entry) => {
      policyVector[moveToIndex(entry.move as Move)] = entry.probability;
    });

    currentGameSamples.push({
      fen: game.fen(),
      policy: policyVector,
      player: 'w',
      value: 0
    });

    const movesWithProbs: MoveProbability[] = mcts.policy
      .map((entry) => ({
        san: entry.move.san ?? '',
        probability: entry.probability,
        isBest: false,
        move: serializeMove(entry.move as Move)
      }))
      .sort((a, b) => b.probability - a.probability);

    if (movesWithProbs.length > 0) movesWithProbs[0].isBest = true;
    topMoves = movesWithProbs.slice(0, 10);
    heatmap = movesWithProbs.map((mp) => ({
      square: mp.move.to,
      intensity: mp.probability
    }));

    currentMetrics = {
      ...currentMetrics,
      entropy: computeEntropy(mcts.policy.map((entry) => entry.probability))
    };

    const sampledIndex = sampleFromPolicy(mcts.policy.map((entry) => entry.probability));
    selectedMove = (mcts.policy[sampledIndex]?.move as Move) ?? (mcts.move as Move | null);
  } else {
    const mcts = await runMctsWithIndicator(game, model, MCTS_DIFFICULTY[difficulty]);
    selectedMove = mcts.move as Move | null;
  }

  const applyMove = (move: Move): Move | null => {
    try {
      const applied = game.move({
        from: move.from,
        to: move.to,
        promotion: move.promotion
      });
      return applied ?? null;
    } catch (err) {
      console.error('Invalid move attempted:', move, err);
      return null;
    }
  };

  if (selectedMove) {
    const applied = applyMove(selectedMove);
    if (!applied) {
      console.warn('Move rejected by engine:', selectedMove);
      const fallbackMoves = game.moves({ verbose: true }) as Move[];
      if (fallbackMoves.length > 0) {
        const fallback = fallbackMoves[Math.floor(Math.random() * fallbackMoves.length)];
        const fallbackApplied = applyMove(fallback);
        if (fallbackApplied) {
          console.warn('Applied fallback random move:', fallback);
          moveHistory = [...moveHistory, fallbackApplied.san];
        }
      }
    } else {
      moveHistory = [...moveHistory, applied.san];
    }
  }

  postState();

  if (game.isGameOver()) {
    const gameCount = currentMetrics.gamesPlayed + 1;
    let outcome = 0;
    if (game.isCheckmate()) {
      outcome = game.turn() === 'w' ? -1 : 1;
    }

    const finalizedSamples = currentGameSamples.map((sample) => ({
      ...sample,
      value: sample.player === 'w' ? outcome : -outcome
    }));
    replayBuffer.push(...finalizedSamples);
    currentGameSamples = [];

    if (replayBuffer.length > MAX_REPLAY_BUFFER) {
      replayBuffer.splice(0, replayBuffer.length - MAX_REPLAY_BUFFER);
    }

    let policyLoss = currentMetrics.policyLoss;
    let valueLoss = currentMetrics.valueLoss;

    const effectiveBatchSize = Math.min(BATCH_SIZE, replayBuffer.length);
    if (effectiveBatchSize > 0) {
      const batch: TrainingSample[] = [];
      for (let i = 0; i < effectiveBatchSize; i++) {
        const sampleIndex = Math.floor(Math.random() * replayBuffer.length);
        batch.push(replayBuffer[sampleIndex]);
      }

      const inputTensors = batch.map((sample) => boardToTensor(new Chess(sample.fen)));
      const stateTensor = tf.concat(inputTensors, 0);
      inputTensors.forEach((t) => t.dispose());
      const policyTensor = tf.tensor2d(batch.map((s) => s.policy), [batch.length, POLICY_OUTPUT_SIZE]);
      const valueTargets = batch.map((s) => s.value);
      const valueTensor = tf.tensor2d(valueTargets, [batch.length, 1]);

      const history = await model.fit(stateTensor, [policyTensor, valueTensor], {
        batchSize: Math.min(BATCH_SIZE, batch.length),
        epochs: 1,
        verbose: 0
      });

      stateTensor.dispose();
      policyTensor.dispose();
      valueTensor.dispose();

      const policyHistory = history.history['policy_head_loss'] ?? history.history['loss'];
      const valueHistory = history.history['value_head_loss'];
      if (policyHistory && policyHistory.length > 0) policyLoss = Number(policyHistory[0]);
      if (valueHistory && valueHistory.length > 0) valueLoss = Number(valueHistory[0]);
    }

    const newWinRate =
      (currentMetrics.winRate * currentMetrics.gamesPlayed + (outcome === 1 ? 1 : 0)) / gameCount;

    const newMetrics: TrainingMetrics = {
      epoch: currentMetrics.epoch + 1,
      gamesPlayed: gameCount,
      policyLoss,
      valueLoss,
      winRate: newWinRate,
      entropy: currentMetrics.entropy
    };

    currentMetrics = newMetrics;
    metricsHistory = [...metricsHistory, newMetrics].slice(-50);

    postState();

    setTimeout(() => {
      game.reset();
      moveHistory = [];
      postState();
    }, 1000);
  }
};

const scheduleLoop = () => {
  if (!isTraining) return;
  const thinkingTime = Math.floor(Math.random() * 500) + 100;
  loopTimeout = setTimeout(async () => {
    try {
      await stepTraining();
    } catch (err) {
      console.error('Training loop error:', err);
    }
    scheduleLoop();
  }, thinkingTime) as unknown as number;
};

const stopLoop = () => {
  if (loopTimeout !== null) {
    clearTimeout(loopTimeout);
    loopTimeout = null;
  }
};

const startTraining = () => {
  if (!model || modelStatus !== 'ready') return;
  if (isTraining) return;
  isTraining = true;
  scheduleLoop();
};

const stopTraining = () => {
  isTraining = false;
  stopLoop();
  setThinking(false);
};

const initModel = async (baseModelUrl: string) => {
  modelStatus = 'loading';
  modelError = undefined;
  postStatus();

  try {
    await tf.ready();
    let newModel: tf.LayersModel;
    try {
      const loaded = await tf.loadLayersModel(baseModelUrl);
      newModel = compileTinyZeroModel(loaded);
      console.info(`Loaded base model from ${baseModelUrl}`);
    } catch (loadErr) {
      newModel = createTinyZeroModel();
      console.warn('Base model not found; using fresh weights.', loadErr);
    }
    model = newModel;
    modelStatus = 'ready';
    postStatus();
    postState();
  } catch (err) {
    console.error('Model init failed:', err);
    modelStatus = 'error';
    modelError = err instanceof Error ? err.message : 'Unknown error';
    postStatus();
  }
};

ctx.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  switch (message.type) {
    case 'init':
      initModel(message.baseModelUrl);
      break;
    case 'start':
      startTraining();
      break;
    case 'stop':
      stopTraining();
      break;
    case 'reset':
      stopTraining();
      replayBuffer.splice(0, replayBuffer.length);
      currentGameSamples = [];
      metricsHistory = [];
      currentMetrics = { ...INITIAL_METRICS };
      resetGameState();
      break;
    case 'setDifficulty':
      difficulty = message.difficulty;
      break;
    default:
      break;
  }
};
