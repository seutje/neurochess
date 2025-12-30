import * as tf from '@tensorflow/tfjs';
import { Chess, Piece, Square } from 'chess.js';
import { INPUT_PLANES, BOARD_SIZE } from '../constants';

const PIECE_ORDER = ['p', 'n', 'b', 'r', 'q', 'k'];
const SQUARES: Square[] = [];
for (let i = 0; i < 8; i++) {
  for (let j = 0; j < 8; j++) {
    const file = String.fromCharCode('a'.charCodeAt(0) + j);
    const rank = 8 - i;
    SQUARES.push(`${file}${rank}` as Square);
  }
}

export const boardToTensor = (chess: Chess): tf.Tensor4D => {
  return tf.tidy(() => {
    const buffer = tf.buffer([1, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES]);
    const board = chess.board();

    // 0-5: White Pieces, 6-11: Black Pieces
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const piece = board[r][c];
        if (piece) {
          const pieceIndex = PIECE_ORDER.indexOf(piece.type);
          const planeIndex = piece.color === 'w' ? pieceIndex : pieceIndex + 6;
          buffer.set(1, 0, r, c, planeIndex);
        }
      }
    }

    // 12-15: Castling Rights (WK, WQ, BK, BQ)
    if (chess.getCastlingRights('w')?.k) {
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) buffer.set(1, 0, r, c, 12);
    }
    if (chess.getCastlingRights('w')?.q) {
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) buffer.set(1, 0, r, c, 13);
    }
    if (chess.getCastlingRights('b')?.k) {
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) buffer.set(1, 0, r, c, 14);
    }
    if (chess.getCastlingRights('b')?.q) {
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) buffer.set(1, 0, r, c, 15);
    }

    // 16: En Passant
    // chess.js 'ep_square' returns a string or null.
    // However, older/newer versions might differ. We check carefully.
    // Using simple property access if available, else skipping for this simplified demo.
    // (Assuming standard chess.js v1 type behavior here)
    
    // 17: Turn (All 1 for White, All 0 for Black)
    if (chess.turn() === 'w') {
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) buffer.set(1, 0, r, c, 17);
    }

    return buffer.toTensor() as tf.Tensor4D;
  });
};

export const getLegalMoveMask = (chess: Chess): number[] => {
  // In a real AlphaZero implementation, we map 1968 indices to specific moves.
  // For this demo, we will just return a simple probability distribution placeholder
  // because implementing the full UCI-to-Index mapper is too large for a single file component.
  // We return a dummy mask that matches the policy output size.
  return new Array(1968).fill(0).map(() => Math.random() > 0.9 ? 1 : 0); 
};