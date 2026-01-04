import { Chess, Move } from 'chess.js';
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import path from 'node:path';
import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { compileTinyZeroModel, createTinyZeroModel } from '../services/model';
import { boardToTensor } from '../services/tensorUtils';
import { runMcts } from '../services/mcts';
import { moveToIndex } from '../services/moveEncoding';
import {
  BATCH_SIZE,
  MAX_REPLAY_BUFFER,
  MIN_REPLAY_START,
  POLICY_LABEL_SMOOTHING,
  POLICY_OUTPUT_SIZE,
  TRAINING_DIRICHLET_ALPHA,
  TRAINING_DIRICHLET_EPSILON,
  TRAINING_MCTS_SIMULATIONS,
  VALUE_TARGET_OUTCOME_WEIGHT,
  MAX_GAME_MOVES
} from '../constants';
import type { MctsConfig, SparsePolicyTarget, TrainingSample } from '../types';

type TrainOptions = {
  games: number;
  outDir: string;
  trainSimulations: number;
  opponentSimulations: number;
  batchSize: number;
};

type CheckpointMeta = {
  gamesPlayed: number;
  winRate: number;
  policyLoss: number;
  valueLoss: number;
  savedAt: string;
};

const CHECKPOINT_INTERVAL_RATIO = 0.1;
const MEMORY_LOG_INTERVAL_RATIO = 0.01;

const args = process.argv.slice(2);

const readArg = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  const direct = args.find((arg) => arg.startsWith(prefix));
  if (direct) return direct.slice(prefix.length);
  const index = args.indexOf(`--${name}`);
  if (index !== -1 && args[index + 1]) return args[index + 1];
  const envKey = `npm_config_${name.replace(/-/g, '_')}`;
  const envValue = process.env[envKey];
  if (envValue) return envValue;
  return fallback;
};

const readNumber = (name: string, fallback: number, aliases: string[] = []): number => {
  const raw = readArg(name, '');
  if (!raw) {
    for (const alias of aliases) {
      const aliasRaw = readArg(alias, '');
      if (aliasRaw) {
        const aliasValue = Number(aliasRaw);
        return Number.isFinite(aliasValue) ? aliasValue : fallback;
      }
    }
  }
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
};

const showHelp = args.includes('--help') || args.includes('-h');
if (showHelp) {
  console.log(`Usage: npm run train:base -- [options]

Options:
  --games <n>             Number of self-play games (default: 10)
  --train-sims <n>        MCTS simulations for training side (default: ${TRAINING_MCTS_SIMULATIONS})
  --training-sims <n>     Alias for --train-sims
  --opponent-sims <n>     MCTS simulations for opponent (default: 80)
  --batch-size <n>        Batch size per update (default: ${BATCH_SIZE})
  --out <dir>             Output directory (default: public/models/base)
  --reset                 Reset to a fresh model instead of loading the last base model
  --resume                Resume from the latest checkpoint in the output directory
  --help, -h              Show this help
`);
  process.exit(0);
}

const positionalNumbers = args
  .filter((arg) => !arg.startsWith('--'))
  .map((arg) => Number(arg))
  .filter((value) => Number.isFinite(value));

const readPositionalNumber = (index: number): number | null => {
  const value = positionalNumbers[index];
  return Number.isFinite(value) ? value : null;
};

const options: TrainOptions = {
  games: Math.max(1, Math.floor(readNumber('games', readPositionalNumber(0) ?? 10))),
  outDir: readArg('out', 'public/models/base'),
  trainSimulations: Math.max(
    1,
    Math.floor(readNumber('train-sims', readPositionalNumber(1) ?? TRAINING_MCTS_SIMULATIONS, ['training-sims']))
  ),
  opponentSimulations: Math.max(1, Math.floor(readNumber('opponent-sims', readPositionalNumber(2) ?? 80))),
  batchSize: Math.max(1, Math.floor(readNumber('batch-size', BATCH_SIZE)))
};
const resetModel = args.includes('--reset');
const resumeTraining = args.includes('--resume');

const sampleFromPolicy = (policy: number[]): number => {
  let threshold = Math.random();
  for (let i = 0; i < policy.length; i++) {
    threshold -= policy[i];
    if (threshold <= 0) return i;
  }
  return policy.length - 1;
};

const applyMove = (game: Chess, move: Move): boolean => {
  try {
    const applied = game.move({
      from: move.from,
      to: move.to,
      promotion: move.promotion
    });
    return Boolean(applied);
  } catch (err) {
    console.warn('Invalid move attempted:', move, err);
    return false;
  }
};

const buildPolicyTarget = (
  game: Chess,
  policy: { move: Move; probability: number }[]
): SparsePolicyTarget => {
  const legalMoves = game.moves({ verbose: true }) as Move[];
  if (legalMoves.length === 0) return { indices: [], probs: [] };

  const base = new Map<number, number>();
  policy.forEach((entry) => {
    base.set(moveToIndex(entry.move), entry.probability);
  });

  const smooth =
    POLICY_LABEL_SMOOTHING > 0 ? POLICY_LABEL_SMOOTHING / legalMoves.length : 0;
  const scale = POLICY_LABEL_SMOOTHING > 0 ? 1 - POLICY_LABEL_SMOOTHING : 1;

  const indices: number[] = [];
  const probs: number[] = [];
  for (const move of legalMoves) {
    const index = moveToIndex(move);
    const baseProb = base.get(index) ?? 0;
    indices.push(index);
    probs.push(baseProb * scale + smooth);
  }

  return { indices, probs };
};

const buildPolicyTensor = (batch: TrainingSample[]): tf.Tensor2D => {
  const data = new Float32Array(batch.length * POLICY_OUTPUT_SIZE);
  batch.forEach((sample, row) => {
    const offset = row * POLICY_OUTPUT_SIZE;
    for (let i = 0; i < sample.policy.indices.length; i++) {
      const index = sample.policy.indices[i] ?? 0;
      const prob = sample.policy.probs[i] ?? 0;
      data[offset + index] = prob;
    }
  });
  return tf.tensor2d(data, [batch.length, POLICY_OUTPUT_SIZE]);
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

const getOpponent = (color: 'w' | 'b'): 'w' | 'b' => (color === 'w' ? 'b' : 'w');

const hasOnlyKing = (gameState: Chess, color: 'w' | 'b'): boolean => {
  let count = 0;
  for (const row of gameState.board()) {
    for (const piece of row) {
      if (!piece || piece.color !== color) continue;
      count += 1;
      if (count > 1) return false;
      if (piece.type !== 'k') return false;
    }
  }
  return count === 1;
};

const getForfeitOutcome = (gameState: Chess): number | null => {
  const turn = gameState.turn();
  if (gameState.moves().length === 0) {
    return getOpponent(turn) === 'w' ? 1 : -1;
  }
  if (hasOnlyKing(gameState, turn)) {
    return getOpponent(turn) === 'w' ? 1 : -1;
  }
  const other = getOpponent(turn);
  if (hasOnlyKing(gameState, other)) {
    return getOpponent(other) === 'w' ? 1 : -1;
  }
  return null;
};

const evaluateValue = async (game: Chess, model: tf.LayersModel): Promise<number> => {
  const tensor = boardToTensor(game);
  const valueTensor = tf.tidy(() => {
    const prediction = model.predict(tensor) as tf.Tensor[];
    return prediction[1] as tf.Tensor;
  });
  const valueData = await valueTensor.data();
  tensor.dispose();
  valueTensor.dispose();
  return valueData[0] ?? 0;
};

const saveModel = async (model: tf.LayersModel, outDir: string) => {
  const resolvedOut = path.resolve(process.cwd(), outDir);
  await mkdir(resolvedOut, { recursive: true });

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const weights = artifacts.weightData
        ? Buffer.from(artifacts.weightData)
        : Buffer.alloc(0);
      const weightSpecs = artifacts.weightSpecs ?? [];

      const modelJson = {
        format: artifacts.format ?? 'layers-model',
        generatedBy: artifacts.generatedBy ?? 'TensorFlow.js',
        convertedBy: artifacts.convertedBy ?? null,
        modelTopology: artifacts.modelTopology,
        trainingConfig: artifacts.trainingConfig,
        userDefinedMetadata: artifacts.userDefinedMetadata,
        weightsManifest: [
          {
            paths: ['weights.bin'],
            weights: weightSpecs
          }
        ]
      };

      await writeFile(path.join(resolvedOut, 'model.json'), JSON.stringify(modelJson, null, 2));
      await writeFile(path.join(resolvedOut, 'weights.bin'), weights);

      const modelTopologyJson = artifacts.modelTopology
        ? JSON.stringify(artifacts.modelTopology)
        : null;
      const modelArtifactsInfo: tf.io.ModelArtifactsInfo = {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelTopologyJson ? Buffer.byteLength(modelTopologyJson) : 0,
        weightDataBytes: weights.byteLength
      };

      return { modelArtifactsInfo };
    })
  );
};

const formatMemoryMb = (bytes: number): string => (bytes / (1024 * 1024)).toFixed(1);

const logMemoryUsage = (label: string) => {
  const usage = process.memoryUsage();
  console.log(
    `${label} | rss ${formatMemoryMb(usage.rss)}MB | heap ${formatMemoryMb(usage.heapUsed)}/${formatMemoryMb(
      usage.heapTotal
    )}MB | external ${formatMemoryMb(usage.external)}MB | arrayBuffers ${formatMemoryMb(
      usage.arrayBuffers ?? 0
    )}MB`
  );
};

const saveCheckpoint = async (
  model: tf.LayersModel,
  outDir: string,
  meta: Omit<CheckpointMeta, 'savedAt'>
) => {
  const checkpointRoot = path.join(outDir, 'checkpoints');
  const checkpointDir = path.join(checkpointRoot, `game-${meta.gamesPlayed}`);
  await saveModel(model, checkpointDir);
  const fullMeta: CheckpointMeta = { ...meta, savedAt: new Date().toISOString() };
  await writeFile(path.join(checkpointDir, 'checkpoint.json'), JSON.stringify(fullMeta, null, 2));
  await writeFile(path.join(checkpointRoot, 'latest.json'), JSON.stringify(fullMeta, null, 2));
  console.log(`Checkpoint saved to ${checkpointDir}`);
};

const loadLatestCheckpoint = async (
  outDir: string
): Promise<{ model: tf.LayersModel; meta: CheckpointMeta | null; gamesPlayed: number } | null> => {
  const checkpointRoot = path.resolve(process.cwd(), outDir, 'checkpoints');
  let entries: string[] = [];
  try {
    entries = await readdir(checkpointRoot);
  } catch (err) {
    return null;
  }
  let bestGame = -1;
  let bestDir: string | null = null;
  for (const entry of entries) {
    const match = /^game-(\d+)$/.exec(entry);
    if (!match) continue;
    const gameCount = Number(match[1]);
    if (!Number.isFinite(gameCount) || gameCount <= bestGame) continue;
    bestGame = gameCount;
    bestDir = entry;
  }
  if (!bestDir || bestGame < 0) return null;
  const checkpointDir = path.join(checkpointRoot, bestDir);
  const modelPath = path.join(checkpointDir, 'model.json');
  await access(modelPath);
  const loaded = await tf.loadLayersModel(`file://${modelPath}`);
  const model = compileTinyZeroModel(loaded);
  let meta: CheckpointMeta | null = null;
  try {
    const raw = await readFile(path.join(checkpointDir, 'checkpoint.json'), 'utf8');
    meta = JSON.parse(raw) as CheckpointMeta;
  } catch (err) {
    meta = null;
  }
  return { model, meta, gamesPlayed: bestGame };
};

const trainBaseModel = async () => {
  console.log('Training TinyZero base model...');
  console.log(`Games: ${options.games}`);
  console.log(`Training sims: ${options.trainSimulations}`);
  console.log(`Opponent sims: ${options.opponentSimulations}`);
  console.log(`Batch size: ${options.batchSize}`);
  console.log(`Output: ${options.outDir}`);

  try {
    await tf.setBackend('tensorflow');
  } catch (err) {
    console.warn('TensorFlow backend unavailable, falling back to CPU:', err);
    await tf.setBackend('cpu');
  }
  await tf.ready();

  const resolvedOutDir = path.resolve(process.cwd(), options.outDir);
  const modelPath = path.join(resolvedOutDir, 'model.json');
  let model: tf.LayersModel;
  let gamesPlayed = 0;
  let winRate = 0;
  let policyLoss = 2.5;
  let valueLoss = 1.0;
  if (resetModel) {
    console.log('Reset flag detected; starting from a fresh model.');
    model = createTinyZeroModel();
  } else if (resumeTraining) {
    const checkpoint = await loadLatestCheckpoint(options.outDir);
    if (checkpoint) {
      model = checkpoint.model;
      gamesPlayed = checkpoint.meta?.gamesPlayed ?? checkpoint.gamesPlayed;
      winRate = checkpoint.meta?.winRate ?? winRate;
      policyLoss = checkpoint.meta?.policyLoss ?? policyLoss;
      valueLoss = checkpoint.meta?.valueLoss ?? valueLoss;
      console.log(`Resumed from checkpoint at game ${gamesPlayed}.`);
      logMemoryUsage('Memory after resume');
    } else {
      console.warn('No checkpoint found; falling back to base model load.');
      try {
        await access(modelPath);
        const loaded = await tf.loadLayersModel(`file://${modelPath}`);
        model = compileTinyZeroModel(loaded);
        console.log(`Loaded base model from ${options.outDir}`);
      } catch (err) {
        console.warn('Base model not found; using fresh weights.', err);
        model = createTinyZeroModel();
      }
    }
  } else {
    try {
      await access(modelPath);
      const loaded = await tf.loadLayersModel(`file://${modelPath}`);
      model = compileTinyZeroModel(loaded);
      console.log(`Loaded base model from ${options.outDir}`);
    } catch (err) {
      console.warn('Base model not found; using fresh weights.', err);
      model = createTinyZeroModel();
    }
  }

  const trainingConfig: MctsConfig = {
    simulations: options.trainSimulations,
    cPuct: 1.5,
    temperature: 1.0,
    dirichletAlpha: TRAINING_DIRICHLET_ALPHA,
    dirichletEpsilon: TRAINING_DIRICHLET_EPSILON
  };

  const opponentConfig: MctsConfig = {
    simulations: options.opponentSimulations,
    cPuct: 1.2,
    temperature: 1.1,
    useHeuristic: true
  };

  const replayBuffer: TrainingSample[] = [];
  const start = Date.now();

  const maxMovesPerGame = MAX_GAME_MOVES;
  const checkpointEvery = Math.max(1, Math.floor(options.games * CHECKPOINT_INTERVAL_RATIO));
  const memoryLogEvery = Math.max(1, Math.floor(options.games * MEMORY_LOG_INTERVAL_RATIO));
  if (gamesPlayed > 0 && gamesPlayed >= options.games) {
    console.log(`Already at ${gamesPlayed} games; nothing to do.`);
    return;
  }

  while (gamesPlayed < options.games) {
    const game = new Chess();
    const currentGameSamples: TrainingSample[] = [];
    let movesPlayed = 0;
    let forfeitOutcome: number | null = null;

    while (!game.isGameOver() && movesPlayed < maxMovesPerGame) {
      const forfeitCheck = getForfeitOutcome(game);
      if (forfeitCheck !== null) {
        forfeitOutcome = forfeitCheck;
        break;
      }
      let selectedMove: Move | null = null;

      if (game.turn() === 'w') {
        const mcts = await runMcts(game, model, trainingConfig, true);
        const policyVector = buildPolicyTarget(game, mcts.policy as { move: Move; probability: number }[]);
        currentGameSamples.push({
          fen: game.fen(),
          policy: policyVector,
          player: 'w',
          value: 0,
          bootstrapValue: mcts.value
        });

        const sampledIndex = sampleFromPolicy(mcts.policy.map((entry) => entry.probability));
        selectedMove = mcts.policy[sampledIndex]?.move ?? mcts.move;
      } else {
        const mcts = await runMcts(game, model, opponentConfig);
        const policyVector = buildPolicyTarget(game, mcts.policy as { move: Move; probability: number }[]);
        currentGameSamples.push({
          fen: game.fen(),
          policy: policyVector,
          player: 'b',
          value: 0,
          bootstrapValue: mcts.value
        });
        selectedMove = mcts.move;
      }

      let moved = false;
      if (selectedMove) {
        const applied = applyMove(game, selectedMove);
        if (!applied) {
          const fallbackMoves = game.moves({ verbose: true }) as Move[];
          if (fallbackMoves.length > 0) {
            const fallback = fallbackMoves[Math.floor(Math.random() * fallbackMoves.length)];
            if (applyMove(game, fallback)) {
              moved = true;
            }
          }
        } else {
          moved = true;
        }
      }
      if (!moved) break;
      movesPlayed += 1;
    }

    const gameCount = gamesPlayed + 1;
    let outcomeForWhite = 0;
    if (forfeitOutcome !== null) {
      outcomeForWhite = forfeitOutcome;
    } else if (movesPlayed >= maxMovesPerGame) {
      outcomeForWhite = computeMaterialOutcome(game);
    } else if (game.isCheckmate()) {
      outcomeForWhite = game.turn() === 'w' ? -1 : 1;
    } else if (game.isDraw()) {
      outcomeForWhite = 0;
    } else {
      const endValue = await evaluateValue(game, model);
      outcomeForWhite = game.turn() === 'w' ? endValue : -endValue;
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
    if (replayBuffer.length > MAX_REPLAY_BUFFER) {
      replayBuffer.splice(0, replayBuffer.length - MAX_REPLAY_BUFFER);
    }

    if (replayBuffer.length >= MIN_REPLAY_START) {
      const effectiveBatchSize = Math.min(options.batchSize, replayBuffer.length);
      const batch = sampleReplayBatch(replayBuffer, effectiveBatchSize);

      const inputTensors = batch.map((sample) => boardToTensor(new Chess(sample.fen)));
      const stateTensor = tf.concat(inputTensors, 0);
      inputTensors.forEach((t) => t.dispose());
      const policyTensor = buildPolicyTensor(batch);
      const valueTargets = batch.map((s) => s.value);
      const valueTensor = tf.tensor2d(valueTargets, [batch.length, 1]);

      const history = await model.fit(stateTensor, [policyTensor, valueTensor], {
        batchSize: Math.min(options.batchSize, batch.length),
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

    winRate = (winRate * gamesPlayed + (outcomeForWhite === 1 ? 1 : 0)) / gameCount;
    gamesPlayed = gameCount;

    console.log(
      `Game ${gamesPlayed}/${options.games} | winRate ${(winRate * 100).toFixed(1)}% | policyLoss ${policyLoss.toFixed(4)} | valueLoss ${valueLoss.toFixed(4)}`
    );
    if (gamesPlayed % memoryLogEvery === 0) {
      logMemoryUsage(`Memory after game ${gamesPlayed}`);
    }
    if (gamesPlayed % checkpointEvery === 0) {
      await saveCheckpoint(model, options.outDir, {
        gamesPlayed,
        winRate,
        policyLoss,
        valueLoss
      });
    }
  }

  await saveModel(model, options.outDir);

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`Saved base model to ${options.outDir} (${elapsed}s).`);
};

trainBaseModel().catch((err) => {
  console.error('Training failed:', err);
  process.exit(1);
});
