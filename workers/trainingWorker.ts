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
  BATCH_SIZE,
  MIN_REPLAY_START,
  POLICY_LABEL_SMOOTHING,
  VALUE_TARGET_OUTCOME_WEIGHT,
  MAX_GAME_MOVES,
  INPUT_PLANES
} from '../constants';
import type {
  TrainingMetrics,
  MoveProbability,
  HeatmapSquare,
  MctsDifficulty,
  MctsConfig,
  TrainingSample,
  MoveLike,
  PerformanceStats
} from '../types';

const INITIAL_METRICS: TrainingMetrics = {
  epoch: 0,
  gamesPlayed: 0,
  policyLoss: 2.5,
  valueLoss: 1.0,
  winRate: 0,
  entropy: 4.5
};

const MCTS_DIFFICULTY: Record<MctsDifficulty, MctsConfig> = {
  easy: { simulations: 80, cPuct: 1.2, temperature: 1.1, useHeuristic: true, logSimTiming: true },
  medium: { simulations: 200, cPuct: 1.4, temperature: 0.8, useHeuristic: true, logSimTiming: true },
  hard: { simulations: 600, cPuct: 1.6, temperature: 0.4, useHeuristic: true, logSimTiming: true }
};

const TRAINING_CONFIG: MctsConfig = {
  simulations: TRAINING_MCTS_SIMULATIONS,
  cPuct: 1.5,
  temperature: 1.0,
  dirichletAlpha: TRAINING_DIRICHLET_ALPHA,
  dirichletEpsilon: TRAINING_DIRICHLET_EPSILON,
  logSimTiming: true
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
  backend?: string;
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
  perfStats: PerformanceStats;
};

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

let model: tf.LayersModel | null = null;
let modelStatus: WorkerStatus = 'loading';
let modelError: string | undefined;
let backend: string | undefined;

let difficulty: MctsDifficulty = 'easy';
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
let movesPlayed = 0;
let perfStats: PerformanceStats = {
  lastMctsMs: 0,
  avgMctsMs: 0,
  mctsRuns: 0,
  lastStepMs: 0,
  avgStepMs: 0,
  steps: 0,
  lastReadbackMs: 0,
  avgReadbackMs: 0,
  readbacks: 0
};
const perfTotals = {
  mctsMs: 0,
  stepMs: 0,
  readbackMs: 0
};

const postStatus = () => {
  const payload: StatusPayload = {
    type: 'status',
    status: modelStatus,
    error: modelError,
    backend
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
    isMctsThinking,
    perfStats
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

const buildPolicyTarget = (game: Chess, policy: { move: Move; probability: number }[]): number[] => {
  const target = new Array(POLICY_OUTPUT_SIZE).fill(0);
  policy.forEach((entry) => {
    target[moveToIndex(entry.move)] = entry.probability;
  });

  if (POLICY_LABEL_SMOOTHING > 0) {
    const legalMoves = game.moves({ verbose: true }) as Move[];
    const legalCount = legalMoves.length;
    if (legalCount > 0) {
      const smooth = POLICY_LABEL_SMOOTHING / legalCount;
      const scale = 1 - POLICY_LABEL_SMOOTHING;
      for (const move of legalMoves) {
        const index = moveToIndex(move);
        target[index] = target[index] * scale + smooth;
      }
    }
  }

  return target;
};

const sampleReplayBatch = (buffer: TrainingSample[], batchSize: number): TrainingSample[] => {
  if (buffer.length <= batchSize) return buffer.slice();
  const indices = Array.from({ length: buffer.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, batchSize).map((index) => buffer[index]);
};

const computeEntropy = (policy: number[]): number => {
  return policy.reduce((acc, p) => (p > 0 ? acc - p * Math.log(p) : acc), 0);
};

const computeMaterialOutcome = (gameState: Chess): number => {
  const pieceValues: Record<string, number> = {
    p: 1,
    n: 3,
    b: 3,
    r: 5,
    q: 9,
    k: 0
  };
  let whiteScore = 0;
  let blackScore = 0;
  for (const row of gameState.board()) {
    for (const piece of row) {
      if (!piece) continue;
      const value = pieceValues[piece.type] ?? 0;
      if (piece.color === 'w') whiteScore += value;
      else blackScore += value;
    }
  }
  if (whiteScore === blackScore) return 0;
  return whiteScore > blackScore ? 1 : -1;
};

const initBackend = async () => {
  tf.enableProdMode();
  const hasOffscreenCanvas = typeof OffscreenCanvas !== 'undefined';
  const preferredBackends = hasOffscreenCanvas ? ['webgl', 'cpu'] : ['cpu'];
  for (const candidate of preferredBackends) {
    if (tf.getBackend() === candidate) break;
    try {
      const ok = await tf.setBackend(candidate);
      if (ok) break;
    } catch (err) {
      console.warn(`Failed to initialize backend ${candidate}.`, err);
    }
  }
  await tf.ready();
  backend = tf.getBackend();
  postStatus();
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
  const start = performance.now();
  try {
    const result = await runMcts(gameState, activeModel, config, addNoise);
    const elapsed = performance.now() - start;
    perfTotals.mctsMs += elapsed;
    perfStats.mctsRuns += 1;
    perfStats.lastMctsMs = elapsed;
    perfStats.avgMctsMs = perfTotals.mctsMs / perfStats.mctsRuns;
    if (result.perf) {
      perfTotals.readbackMs += result.perf.readbackMsTotal;
      perfStats.readbacks += result.perf.readbacks;
      perfStats.lastReadbackMs = result.perf.lastReadbackMs;
      perfStats.avgReadbackMs = perfStats.readbacks > 0 ? perfTotals.readbackMs / perfStats.readbacks : 0;
    }
    return result;
  } finally {
    setThinking(false);
  }
};

const resetGameState = () => {
  game.reset();
  heatmap = [];
  topMoves = [];
  moveHistory = [];
  movesPlayed = 0;
  postState();
};

const stepTraining = async () => {
  if (!model) return;
  if (game.isGameOver() || movesPlayed >= MAX_GAME_MOVES) return;
  const stepStart = performance.now();

  try {
    const turn = game.turn();
    let selectedMove: Move | null = null;

    if (turn === 'w') {
      const mcts = await runMctsWithIndicator(game, model, TRAINING_CONFIG, true);
      const policyVector = buildPolicyTarget(game, mcts.policy as { move: Move; probability: number }[]);

      currentGameSamples.push({
        fen: game.fen(),
        policy: policyVector,
        player: 'w',
        value: 0,
        bootstrapValue: mcts.value
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
      const policyVector = buildPolicyTarget(game, mcts.policy as { move: Move; probability: number }[]);
      currentGameSamples.push({
        fen: game.fen(),
        policy: policyVector,
        player: 'b',
        value: 0,
        bootstrapValue: mcts.value
      });
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
            movesPlayed += 1;
          }
        }
      } else {
        moveHistory = [...moveHistory, applied.san];
        movesPlayed += 1;
      }
    }

    postState();

    if (movesPlayed >= MAX_GAME_MOVES || game.isGameOver()) {
      const gameCount = currentMetrics.gamesPlayed + 1;
      let outcomeForWhite = 0;
      if (movesPlayed >= MAX_GAME_MOVES) {
        outcomeForWhite = computeMaterialOutcome(game);
      } else if (game.isCheckmate()) {
        outcomeForWhite = game.turn() === 'w' ? -1 : 1;
      } else if (game.isDraw()) {
        outcomeForWhite = 0;
      }

      const finalizedSamples = currentGameSamples.map((sample) => {
        const outcomeForPlayer = sample.player === 'w' ? outcomeForWhite : -outcomeForWhite;
        const bootstrapValue = sample.bootstrapValue ?? 0;
        const blendedValue =
          VALUE_TARGET_OUTCOME_WEIGHT * outcomeForPlayer +
          (1 - VALUE_TARGET_OUTCOME_WEIGHT) * bootstrapValue;
        return {
          ...sample,
          value: blendedValue
        };
      });
      replayBuffer.push(...finalizedSamples);
      currentGameSamples = [];

      if (replayBuffer.length > MAX_REPLAY_BUFFER) {
        replayBuffer.splice(0, replayBuffer.length - MAX_REPLAY_BUFFER);
      }

      let policyLoss = currentMetrics.policyLoss;
      let valueLoss = currentMetrics.valueLoss;

      if (replayBuffer.length >= MIN_REPLAY_START) {
        const effectiveBatchSize = Math.min(BATCH_SIZE, replayBuffer.length);
        const batch = sampleReplayBatch(replayBuffer, effectiveBatchSize);

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
        (currentMetrics.winRate * currentMetrics.gamesPlayed + (outcomeForWhite === 1 ? 1 : 0)) / gameCount;

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
        movesPlayed = 0;
        postState();
      }, 1000);
    }
  } finally {
    const elapsed = performance.now() - stepStart;
    perfTotals.stepMs += elapsed;
    perfStats.steps += 1;
    perfStats.lastStepMs = elapsed;
    perfStats.avgStepMs = perfTotals.stepMs / perfStats.steps;
    postState();
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
    await initBackend();
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
    const warmupOutputs = tf.tidy(() => {
      const warmupInput = tf.zeros([1, 8, 8, INPUT_PLANES]);
      return newModel.predict(warmupInput) as tf.Tensor[];
    });
    if (Array.isArray(warmupOutputs)) {
      warmupOutputs.forEach((tensor) => tensor.dispose());
    } else {
      warmupOutputs.dispose();
    }
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
      perfStats = {
        lastMctsMs: 0,
        avgMctsMs: 0,
        mctsRuns: 0,
        lastStepMs: 0,
        avgStepMs: 0,
        steps: 0,
        lastReadbackMs: 0,
        avgReadbackMs: 0,
        readbacks: 0
      };
      perfTotals.mctsMs = 0;
      perfTotals.stepMs = 0;
      perfTotals.readbackMs = 0;
      resetGameState();
      break;
    case 'setDifficulty':
      difficulty = message.difficulty;
      break;
    default:
      break;
  }
};
