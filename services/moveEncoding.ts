import type { Move, Square } from 'chess.js';

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const PROMOTIONS = ['', 'q', 'r', 'b', 'n'] as const;

export const PROMOTION_COUNT = PROMOTIONS.length;
export const POLICY_SIZE = 64 * 64 * PROMOTION_COUNT;

export const squareToIndex = (square: Square): number => {
  const file = square[0];
  const rank = Number(square[1]);
  const fileIndex = FILES.indexOf(file);
  return (rank - 1) * 8 + fileIndex;
};

export const indexToSquare = (index: number): Square => {
  const rank = Math.floor(index / 8) + 1;
  const file = FILES[index % 8];
  return `${file}${rank}` as Square;
};

export const moveToIndex = (move: Move): number => {
  const fromIndex = squareToIndex(move.from);
  const toIndex = squareToIndex(move.to);
  const promotionIndex = PROMOTIONS.indexOf((move.promotion ?? '') as typeof PROMOTIONS[number]);
  const promo = promotionIndex === -1 ? 0 : promotionIndex;
  return (fromIndex * 64 + toIndex) * PROMOTION_COUNT + promo;
};

export const indexToMoveKey = (index: number): string => {
  const promo = index % PROMOTION_COUNT;
  const base = Math.floor(index / PROMOTION_COUNT);
  const fromIndex = Math.floor(base / 64);
  const toIndex = base % 64;
  const from = indexToSquare(fromIndex);
  const to = indexToSquare(toIndex);
  const promotion = PROMOTIONS[promo];
  return `${from}${to}${promotion}`;
};
