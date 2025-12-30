import { Chess, Move } from 'chess.js';
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import path from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { createTinyZeroModel } from '../services/model';
import { boardToTensor } from '../services/tensorUtils';
import { runMcts } from '../services/mcts';
import { moveToIndex } from '../services/moveEncoding';
import {
  BATCH_SIZE,
  MAX_REPLAY_BUFFER,
  POLICY_OUTPUT_SIZE,
  TRAINING_DIRICHLET_ALPHA,
  TRAINING_DIRICHLET_EPSILON,
  TRAINING_MCTS_SIMULATIONS
} from '../constants';
import type { MctsConfig, TrainingSample } from '../types';

type TrainOptions = {
  games: number;
  outDir: string;
  trainSimulations: number;
  opponentSimulations: number;
  batchSize: number;
};

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

  const model = createTinyZeroModel();

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
    temperature: 1.1
  };

  let gamesPlayed = 0;
  let winRate = 0;
  let policyLoss = 2.5;
  let valueLoss = 1.0;

  const replayBuffer: TrainingSample[] = [];
  const start = Date.now();

  while (gamesPlayed < options.games) {
    const game = new Chess();
    const currentGameSamples: TrainingSample[] = [];

    while (!game.isGameOver()) {
      let selectedMove: Move | null = null;

      if (game.turn() === 'w') {
        const mcts = await runMcts(game, model, trainingConfig, true);
        const policyVector = new Array(POLICY_OUTPUT_SIZE).fill(0);
        mcts.policy.forEach((entry) => {
          policyVector[moveToIndex(entry.move)] = entry.probability;
        });

        currentGameSamples.push({
          fen: game.fen(),
          policy: policyVector,
          player: 'w',
          value: 0
        });

        const sampledIndex = sampleFromPolicy(mcts.policy.map((entry) => entry.probability));
        selectedMove = mcts.policy[sampledIndex]?.move ?? mcts.move;
      } else {
        const mcts = await runMcts(game, model, opponentConfig);
        selectedMove = mcts.move;
      }

      if (selectedMove) {
        const applied = applyMove(game, selectedMove);
        if (!applied) {
          const fallbackMoves = game.moves({ verbose: true }) as Move[];
          if (fallbackMoves.length > 0) {
            const fallback = fallbackMoves[Math.floor(Math.random() * fallbackMoves.length)];
            applyMove(game, fallback);
          }
        }
      }
    }

    const gameCount = gamesPlayed + 1;
    let outcome = 0;
    if (game.isCheckmate()) {
      outcome = game.turn() === 'w' ? -1 : 1;
    }

    const finalizedSamples = currentGameSamples.map((sample) => ({
      ...sample,
      value: sample.player === 'w' ? outcome : -outcome
    }));
    replayBuffer.push(...finalizedSamples);
    if (replayBuffer.length > MAX_REPLAY_BUFFER) {
      replayBuffer.splice(0, replayBuffer.length - MAX_REPLAY_BUFFER);
    }

    if (replayBuffer.length >= options.batchSize) {
      const batch: TrainingSample[] = [];
      for (let i = 0; i < options.batchSize; i++) {
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

    winRate = (winRate * gamesPlayed + (outcome === 1 ? 1 : 0)) / gameCount;
    gamesPlayed = gameCount;

    console.log(
      `Game ${gamesPlayed}/${options.games} | winRate ${(winRate * 100).toFixed(1)}% | policyLoss ${policyLoss.toFixed(4)} | valueLoss ${valueLoss.toFixed(4)}`
    );
  }

  await saveModel(model, options.outDir);

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`Saved base model to ${options.outDir} (${elapsed}s).`);
};

trainBaseModel().catch((err) => {
  console.error('Training failed:', err);
  process.exit(1);
});
