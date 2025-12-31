export const BOARD_SIZE = 8;
export const INPUT_PLANES = 18; // P, N, B, R, Q, K (x2) + Castling(4) + EP(1) + Turn(1)

// Model Architecture Constants
export const FILTERS = 64;
export const RESIDUAL_BLOCKS = 4;
export const POLICY_OUTPUT_SIZE = 64 * 64 * 5; // From-To with promotion channels

// Training Hyperparameters
export const LEARNING_RATE = 0.001;
export const BATCH_SIZE = 512;
export const MCTS_ROLLOUT_DEPTH = 10;
export const TRAINING_MCTS_SIMULATIONS = 600;
export const TRAINING_DIRICHLET_ALPHA = 0.3;
export const TRAINING_DIRICHLET_EPSILON = 0.25;
export const MAX_REPLAY_BUFFER = 4096;
export const MIN_REPLAY_START = 512;
export const POLICY_LABEL_SMOOTHING = 0.1;
export const VALUE_TARGET_OUTCOME_WEIGHT = 0.7;
export const MAX_GAME_MOVES = 200;

// Chart Colors
export const COLORS = {
  policy: '#00f0ff',
  value: '#ff003c',
  entropy: '#00ff9d',
  grid: '#2a2a40',
  text: '#a0a0b0'
};
