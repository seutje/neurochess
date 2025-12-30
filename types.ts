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

import type { Move } from 'chess.js';

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
