import { Chess, Move } from 'chess.js';
import * as tf from '@tensorflow/tfjs';
import { boardToTensor } from './tensorUtils';
import { moveToIndex } from './moveEncoding';
import type { MctsConfig, MctsResult, MctsMoveStats } from '../types';

type MctsEval = {
  policy: Float32Array;
  value: number;
};

class MctsNode {
  prior: number;
  visitCount: number;
  valueSum: number;
  children: Map<string, MctsNode>;
  move: Move | null;

  constructor(prior: number, move: Move | null) {
    this.prior = prior;
    this.visitCount = 0;
    this.valueSum = 0;
    this.children = new Map();
    this.move = move;
  }

  get qValue(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }
}

const getOutcomeValue = (game: Chess): number => {
  if (game.isDraw()) return 0;
  if (game.isCheckmate()) {
    // Side to move is checkmated.
    return -1;
  }
  return 0;
};

const evaluatePosition = async (game: Chess, model: tf.LayersModel): Promise<MctsEval> => {
  const tensor = boardToTensor(game);
  const [policyTensor, valueTensor] = tf.tidy(() => {
    const prediction = model.predict(tensor) as tf.Tensor[];
    return [prediction[0], prediction[1]];
  });
  const policy = await (policyTensor as tf.Tensor).data();
  const value = await (valueTensor as tf.Tensor).data();
  tensor.dispose();
  (policyTensor as tf.Tensor).dispose();
  (valueTensor as tf.Tensor).dispose();
  return { policy: new Float32Array(policy), value: value[0] };
};

const selectChild = (node: MctsNode, cPuct: number): MctsNode => {
  let bestScore = -Infinity;
  let bestChild: MctsNode | null = null;
  const parentVisits = Math.max(1, node.visitCount);

  node.children.forEach((child) => {
    const u = cPuct * child.prior * Math.sqrt(parentVisits) / (1 + child.visitCount);
    const score = child.qValue + u;
    if (score > bestScore) {
      bestScore = score;
      bestChild = child;
    }
  });

  return bestChild ?? Array.from(node.children.values())[0];
};

const addDirichletNoise = (priors: number[], alpha: number, epsilon: number): number[] => {
  const noise = new Array(priors.length).fill(0).map(() => {
    const sample = -Math.log(Math.random());
    return Math.pow(sample, 1 / alpha);
  });
  const sum = noise.reduce((acc, n) => acc + n, 0) || 1;
  return priors.map((p, i) => (1 - epsilon) * p + epsilon * (noise[i] / sum));
};

const buildMoveStats = (game: Chess, root: MctsNode): MctsMoveStats[] => {
  const moves = game.moves({ verbose: true });
  return moves.map((move) => {
    const key = `${move.from}${move.to}${move.promotion ?? ''}`;
    const child = root.children.get(key);
    return {
      move,
      visitCount: child?.visitCount ?? 0,
      prior: child?.prior ?? 0,
      value: child?.qValue ?? 0
    };
  });
};

const pickMoveFromVisits = (stats: MctsMoveStats[], temperature: number): Move | null => {
  if (stats.length === 0) return null;
  if (temperature <= 0) {
    return stats.reduce((best, current) => (current.visitCount > best.visitCount ? current : best)).move;
  }
  const weights = stats.map((stat) => Math.pow(stat.visitCount, 1 / temperature));
  const total = weights.reduce((acc, w) => acc + w, 0);
  let threshold = Math.random() * (total || 1);
  for (let i = 0; i < stats.length; i++) {
    threshold -= weights[i];
    if (threshold <= 0) return stats[i].move;
  }
  return stats[0].move;
};

export const runMcts = async (
  game: Chess,
  model: tf.LayersModel,
  config: MctsConfig,
  trainingNoise = false
): Promise<MctsResult> => {
  const root = new MctsNode(1, null);
  const rootFen = game.fen();

  for (let i = 0; i < config.simulations; i++) {
    const simulation = new Chess(rootFen);
    const path: MctsNode[] = [root];
    let node = root;

    while (node.children.size > 0) {
      const selected = selectChild(node, config.cPuct);
      const move = selected.move;
      if (!move) break;
      simulation.move({ from: move.from, to: move.to, promotion: move.promotion });
      node = selected;
      path.push(node);
    }

    let value: number;
    if (simulation.isGameOver()) {
      value = getOutcomeValue(simulation);
    } else {
      const evalResult = await evaluatePosition(simulation, model);
      const legalMoves = simulation.moves({ verbose: true });
      const priors = legalMoves.map((move) => {
        const index = moveToIndex(move);
        return evalResult.policy[index] ?? 0;
      });
      const sum = priors.reduce((acc, p) => acc + p, 0);
      let normalized = priors.map((p) => (sum > 0 ? p / sum : 1 / priors.length));
      if (trainingNoise && node === root && config.dirichletAlpha && config.dirichletEpsilon) {
        normalized = addDirichletNoise(normalized, config.dirichletAlpha, config.dirichletEpsilon);
      }
      legalMoves.forEach((move, idx) => {
        const key = `${move.from}${move.to}${move.promotion ?? ''}`;
        node.children.set(key, new MctsNode(normalized[idx], move));
      });
      value = evalResult.value;
    }

    for (let p = path.length - 1; p >= 0; p--) {
      const current = path[p];
      current.visitCount += 1;
      current.valueSum += value;
      value = -value;
    }
  }

  const stats = buildMoveStats(game, root);
  const move = pickMoveFromVisits(stats, config.temperature);
  const totalVisits = stats.reduce((acc, stat) => acc + stat.visitCount, 0) || 1;
  const policy = stats.map((stat) => ({
    move: stat.move,
    probability: stat.visitCount / totalVisits,
    visitCount: stat.visitCount
  }));

  return { move, policy };
};
