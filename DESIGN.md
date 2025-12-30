# Design Document: NeuroChess
**Project:** Browser-Based Self-Training Chess AI  
**Date:** October 26, 2023  
**Status:** Draft v1.0

---

## 1. Executive Summary
**NeuroChess-Web** is a client-side web application demonstrating Reinforcement Learning (RL) in the browser. It features a Neural Network (NN) that learns to play chess from scratch by playing against itself. 

The system is designed to run entirely in the user's browser without backend dependencies, utilizing **TensorFlow.js** for GPU acceleration. The training process is visualized in real-time, and the final model is evaluated against a fixed-strength Monte Carlo Tree Search (MCTS) engine with a specific rollout depth.

---

## 2. Neural Network Research & Selection

### 2.1 Constraints
* **Environment:** Browser (Chrome/Firefox/Edge).
* **Hardware:** Consumer-grade GPU (via WebGL/WebGPU).
* **Latency:** Training must be fast enough to show progress within minutes, not days.
* **Memory:** Limited VRAM usage to prevent browser crashes.

### 2.2 Selected Architecture: "TinyZero" (Shallow ResNet)
Research into AlphaZero and Leela Chess Zero (Lc0) indicates that while Transformers are rising in popularity, **Residual Convolutional Neural Networks (ResNets)** remain the most efficient architecture for board games where spatial locality (e.g., pawn structures, diagonal attacks) is critical.

For a browser-based implementation, a full 40-block ResNet is too slow. We will use a "TinyZero" architecture:
* **Type:** Convolutional Neural Network (CNN) with Residual Connections.
* **Depth:** 4 to 5 Residual Blocks.
* **Width:** 64 Filters.
* **Heads:** Dual-headed (Policy & Value).

**Justification:**
1.  **Spatial Awareness:** CNNs naturally handle the 2D grid structure of chess.
2.  **Training Speed:** A shallow ResNet (4 blocks) offers the optimal trade-off between evaluating positions quickly (high games/second) and learning complex tactics.
3.  **Stability:** Residual connections prevent the "vanishing gradient" problem during backpropagation.

---

## 3. Technical Architecture

### 3.1 Tech Stack
* **Core:** React.js or Vue.js (UI Components).
* **ML Engine:** TensorFlow.js (WebGL/WebGPU backend).
* **Game Logic:** `chess.js` (Move validation) or a custom WASM module for speed.
* **State Management:** Redux or Zustand.
* **Concurrency:** Web Workers (Critical to prevent UI freezing).

### 3.2 Threading Model
The application uses a 3-thread architecture:

| Thread | Role | Responsibilities |
| :--- | :--- | :--- |
| **Main Thread** | UI & Rendering | Renders the board, updates charts, handles user clicks. |
| **Worker A** | The "Brain" (NN) | Holds the TF.js model. Executes inference, performs gradient descent (training), and saves weights. |
| **Worker B** | The "Referee" (MCTS) | Generates self-play games using the NN. Runs the benchmark MCTS opponent logic. |

---

## 4. Detailed Component Specifications

### 4.1 Input Representation (The "Tensors")
The board state is converted into an `8x8xN` matrix (Tensor) before being fed to the NN.
* **Input Shape:** `[8, 8, 18]`
* **Planes (18 total):**
    * **0-5:** White Pieces (P, N, B, R, Q, K) - 1s for presence, 0 for empty.
    * **6-11:** Black Pieces.
    * **12-15:** Castling Rights (White K-side, White Q-side, Black K-side, Black Q-side).
    * **16:** En Passant target (if any).
    * **17:** Turn Indicator (All 1s for White, All 0s for Black).

### 4.2 Neural Network Architecture (TensorFlow.js)
```javascript
// Conceptual Model Definition
const input = tf.input({shape: [8, 8, 18]});

// 1. Convolutional Entry
let x = tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same'}).apply(input);
x = tf.layers.batchNormalization().apply(x);
x = tf.layers.reLU().apply(x);

// 2. Residual Tower (4 Blocks)
for (let i = 0; i < 4; i++) {
    let skip = x;
    x = tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same'}).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.reLU().apply(x);
    x = tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same'}).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.add().apply([x, skip]); // The Residual Connection
    x = tf.layers.reLU().apply(x);
}

// 3. Policy Head (Move Probabilities)
let p = tf.layers.conv2d({filters: 2, kernelSize: 1}).apply(x);
p = tf.layers.flatten().apply(p);
p = tf.layers.dense({units: 1968, activation: 'softmax'}).apply(p); // 1968 = approx max moves

// 4. Value Head (Win Probability)
let v = tf.layers.conv2d({filters: 1, kernelSize: 1}).apply(x);
v = tf.layers.flatten().apply(v);
v = tf.layers.dense({units: 64, activation: 'relu'}).apply(v);
v = tf.layers.dense({units: 1, activation: 'tanh'}).apply(v);

const model = tf.model({inputs: input, outputs: [p, v]});
````

### 4.3 The Opponent: MCTS (Depth 10)

The benchmark opponent is a standard MCTS engine constrained to a specific rollout depth to simulate a "mid-level" human calculator.

  * **Selection:** UCB1 (Upper Confidence Bound).
  * **Expansion:** All legal moves.
  * **Simulation (The "Rollout"):**
      * Instead of playing to the end of the game (which is slow), the rollout plays **exactly 10 moves** using a random policy.
      * **Static Evaluation:** At Depth 10, the board is scored using a heuristic function:
          * `Score = Material_Count + Piece_Square_Tables + Mobility_Bonus`
      * This score is backpropagated up the tree.

-----

## 5\. The Training Loop (Self-Play)

1.  **Game Generation:**
      * The NN plays against itself.
      * **Temperature:** For the first 30 moves, moves are chosen probabilistically (to ensure game diversity). After move 30, the greedy (best) move is chosen.
2.  **Data Storage:**
      * `(State, Policy, Value)` tuples are stored in a circular buffer (max size: 2048 games).
3.  **Optimization:**
      * Every `X` games, the "Trainer Worker" samples a batch (size 64 or 128).
      * **Loss Function:** Sum of Mean Squared Error (for Value) and Cross-Entropy Loss (for Policy).
      * Optimizer: SGD with Nesterov Momentum or Adam.

-----

## 6\. UI & Data Visualization

The dashboard serves as the window into the "black box" of the neural network.

### 6.1 Real-Time Charts

  * **Loss History:** Line chart tracking `Policy Loss` and `Value Loss` per epoch.
      * *Goal:* Should trend downward.
  * **Elo Estimation:** Line chart simulating Elo rating based on win/loss ratio against the MCTS baseline.
  * **Entropy:** Measures how "confused" the network is. High entropy = unsure of move; Low entropy = confident.

### 6.2 The Board

  * **Heatmap Overlay:** When the NN is thinking, overlay squares with color opacity representing the Policy output (red = bad move, green = predicted best move).
  * **Analysis Panel:** Display the "top 3 moves" the NN considered and their assigned probabilities.
