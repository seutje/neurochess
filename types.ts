import type { Square } from 'chess.js';

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
  move: MoveLike;
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

export interface MoveLike {
  from: Square;
  to: Square;
  promotion?: string;
  san?: string;
}

export interface MctsMoveStats {
  move: MoveLike;
  visitCount: number;
  prior: number;
  value: number;
}

export interface MctsResult {
  move: MoveLike | null;
  policy: {
    move: MoveLike;
    probability: number;
    visitCount: number;
  }[];
  value: number;
}

export interface TrainingSample {
  fen: string;
  policy: number[];
  player: 'w' | 'b';
  value: number;
  bootstrapValue?: number;
}
