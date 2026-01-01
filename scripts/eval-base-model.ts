import { Chess } from 'chess.js';
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import path from 'node:path';
import { access } from 'node:fs/promises';
import { runMcts } from '../services/mcts';
import { MAX_GAME_MOVES, TRAINING_MCTS_SIMULATIONS } from '../constants';
import type { MctsConfig } from '../types';

type EvalOptions = {
  games: number;
  simulations: number;
  temperature: number;
  cPuct: number;
  modelPath: string;
  maxMoves: number;
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

const readNumber = (name: string, fallback: number): number => {
  const raw = readArg(name, '');
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
};

const showHelp = args.includes('--help') || args.includes('-h');
if (showHelp) {
  console.log(`Usage: npm run eval:base -- [options]

Options:
  --games <n>         Number of evaluation games (default: 50)
  --sims <n>          MCTS simulations per move (default: ${TRAINING_MCTS_SIMULATIONS})
  --temperature <n>   MCTS temperature (default: 0)
  --cpuct <n>         MCTS cPuct value (default: 1.5)
  --model <path>      Model directory or model.json (default: public/models/base)
  --max-moves <n>     Max moves before declaring a draw (default: ${MAX_GAME_MOVES})
  --help, -h          Show this help
`);
  process.exit(0);
}

const options: EvalOptions = {
  games: Math.max(1, Math.floor(readNumber('games', 50))),
  simulations: Math.max(1, Math.floor(readNumber('sims', TRAINING_MCTS_SIMULATIONS))),
  temperature: readNumber('temperature', 0),
  cPuct: readNumber('cpuct', 1.5),
  modelPath: readArg('model', 'public/models/base'),
  maxMoves: Math.max(1, Math.floor(readNumber('max-moves', MAX_GAME_MOVES)))
};

const resolveModelPath = (input: string): string => {
  const resolved = path.resolve(process.cwd(), input);
  return resolved.endsWith('.json') ? resolved : path.join(resolved, 'model.json');
};

const evaluateModel = async () => {
  console.log('Evaluating base model with symmetric MCTS settings...');
  console.log(`Games: ${options.games}`);
  console.log(`Simulations: ${options.simulations}`);
  console.log(`Temperature: ${options.temperature}`);
  console.log(`cPuct: ${options.cPuct}`);
  console.log(`Max moves: ${options.maxMoves}`);
  console.log(`Model: ${options.modelPath}`);

  try {
    await tf.setBackend('tensorflow');
  } catch (err) {
    console.warn('TensorFlow backend unavailable, falling back to CPU:', err);
    await tf.setBackend('cpu');
  }
  await tf.ready();

  const modelPath = resolveModelPath(options.modelPath);
  await access(modelPath);
  const model = await tf.loadLayersModel(`file://${modelPath}`);

  const config: MctsConfig = {
    simulations: options.simulations,
    cPuct: options.cPuct,
    temperature: options.temperature,
    useHeuristic: false
  };

  let whiteWins = 0;
  let blackWins = 0;
  let draws = 0;
  const start = Date.now();

  for (let i = 0; i < options.games; i++) {
    const game = new Chess();
    let movesPlayed = 0;

    while (!game.isGameOver() && movesPlayed < options.maxMoves) {
      const mcts = await runMcts(game, model, config);
      const move = mcts.move;
      if (!move) break;
      game.move({ from: move.from, to: move.to, promotion: move.promotion });
      movesPlayed += 1;
    }

    if (game.isCheckmate()) {
      const winner = game.turn() === 'w' ? 'b' : 'w';
      if (winner === 'w') whiteWins += 1;
      else blackWins += 1;
    } else {
      draws += 1;
    }

    const total = i + 1;
    const whiteWinRate = (whiteWins / total) * 100;
    const blackWinRate = (blackWins / total) * 100;
    const drawRate = (draws / total) * 100;
    console.log(
      `Game ${total}/${options.games} | white ${whiteWinRate.toFixed(1)}% | black ${blackWinRate.toFixed(1)}% | draws ${drawRate.toFixed(1)}%`
    );
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(
    `Done in ${elapsed}s. White ${whiteWins}, Black ${blackWins}, Draws ${draws} (Total ${options.games}).`
  );
};

evaluateModel().catch((err) => {
  console.error('Evaluation failed:', err);
  process.exit(1);
});
