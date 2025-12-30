import type { Move } from 'chess.js';

export enum PlayerType {
  HUMAN = 'HUMAN',
  AI_TRAINING = 'AI_TRAINING',
  AI_MCTS = 'AI_MCTS'
}

export interface TrainingMetrics {
  epoch: number;
  gamesPlayed: number;
  policyLoss: number;
  valueLoss: number;
  winRate: number; // vs Random/MCTS
  entropy: number;
}

export interface MoveProbability {
  san: string;
  probability: number;
  isBest: boolean;
  move: Move;
}

export interface HeatmapSquare {
  square: string; // e.g., 'e4'
  intensity: number; // 0 to 1
}

export type PieceSymbol = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type Color = 'w' | 'b';

export type MctsDifficulty = 'easy' | 'medium' | 'hard';

export interface MctsConfig {
  simulations: number;
  cPuct: number;
  temperature: number;
  dirichletAlpha?: number;
  dirichletEpsilon?: number;
}

export interface MctsMoveStats {
  move: Move;
  visitCount: number;
  prior: number;
  value: number;
}

export interface MctsResult {
  move: Move | null;
  policy: {
    move: Move;
    probability: number;
    visitCount: number;
  }[];
}

export interface TrainingSample {
  fen: string;
  policy: number[];
  player: 'w' | 'b';
  value: number;
}
