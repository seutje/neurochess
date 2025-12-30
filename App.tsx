import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chess, Move } from 'chess.js';
import * as tf from '@tensorflow/tfjs';
import { Activity, Brain, Cpu, Play, StopCircle, RefreshCw, Circle } from 'lucide-react';

import { HeatmapBoard } from './components/HeatmapBoard';
import { LossChart, EntropyChart } from './components/Charts';
import { MoveAnalysis } from './components/MoveAnalysis';
import { compileTinyZeroModel, createTinyZeroModel } from './services/model';
import { boardToTensor } from './services/tensorUtils';
import { runMcts } from './services/mcts';
import { moveToIndex } from './services/moveEncoding';
import { TrainingMetrics, MoveProbability, HeatmapSquare, MctsDifficulty, TrainingSample, MctsConfig } from './types';
import {
  POLICY_OUTPUT_SIZE,
  TRAINING_MCTS_SIMULATIONS,
  TRAINING_DIRICHLET_ALPHA,
  TRAINING_DIRICHLET_EPSILON,
  MAX_REPLAY_BUFFER,
  BATCH_SIZE
} from './constants';

const INITIAL_METRICS: TrainingMetrics = {
  epoch: 0,
  gamesPlayed: 0,
  policyLoss: 2.5,
  valueLoss: 1.0,
  winRate: 0.1,
  entropy: 4.5
};

const MCTS_DIFFICULTY: Record<MctsDifficulty, MctsConfig> = {
  easy: { simulations: 80, cPuct: 1.2, temperature: 1.1 },
  medium: { simulations: 200, cPuct: 1.4, temperature: 0.8 },
  hard: { simulations: 600, cPuct: 1.6, temperature: 0.4 }
};

const TRAINING_CONFIG: MctsConfig = {
  simulations: TRAINING_MCTS_SIMULATIONS,
  cPuct: 1.5,
  temperature: 1.0,
  dirichletAlpha: TRAINING_DIRICHLET_ALPHA,
  dirichletEpsilon: TRAINING_DIRICHLET_EPSILON
};

const sampleFromPolicy = (policy: number[]): number => {
  let threshold = Math.random();
  for (let i = 0; i < policy.length; i++) {
    threshold -= policy[i];
    if (threshold <= 0) return i;
  }
  return policy.length - 1;
};

const computeEntropy = (policy: number[]): number => {
  return policy.reduce((acc, p) => (p > 0 ? acc - p * Math.log(p) : acc), 0);
};

const App: React.FC = () => {
  // Game State
  const [game, setGame] = useState(new Chess());
  const [isTraining, setIsTraining] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState<MctsDifficulty>('medium');
  const [isMctsThinking, setIsMctsThinking] = useState(false);
  
  // Metrics & Visuals
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics>(INITIAL_METRICS);
  const [heatmap, setHeatmap] = useState<HeatmapSquare[]>([]);
  const [topMoves, setTopMoves] = useState<MoveProbability[]>([]);
  
  // Refs for loops
  const gameRef = useRef(new Chess());
  const replayBufferRef = useRef<TrainingSample[]>([]);
  const currentGameSamplesRef = useRef<TrainingSample[]>([]);

  const runMctsWithIndicator = useCallback(
    async (gameState: Chess, activeModel: tf.LayersModel, config: MctsConfig, addNoise = false) => {
      setIsMctsThinking(true);
      try {
        return await runMcts(gameState, activeModel, config, addNoise);
      } finally {
        setIsMctsThinking(false);
      }
    },
    []
  );

  // Initialize TF Model
  useEffect(() => {
    const initModel = async () => {
      setModelStatus('loading');
      setModelError(null);
      try {
        await tf.ready();
        let newModel: tf.LayersModel;
        try {
          const baseRoot = new URL(import.meta.env.BASE_URL ?? '/', window.location.origin);
          const baseModelUrl = new URL('models/base/model.json', baseRoot).toString();
          const loaded = await tf.loadLayersModel(baseModelUrl);
          newModel = compileTinyZeroModel(loaded);
          console.info(`Loaded base model from ${baseModelUrl}`);
        } catch (loadErr) {
          newModel = createTinyZeroModel();
          console.warn('Base model not found; using fresh weights.', loadErr);
        }
        setModel(newModel);
        setModelStatus('ready');
        newModel.summary();
        console.log("TinyZero Model Initialized.");
      } catch (err) {
        console.error('Model init failed:', err);
        setModelStatus('error');
        setModelError(err instanceof Error ? err.message : 'Unknown error');
      }
    };
    initModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Core "AI" Logic (Simulated for Web Browser Latency) ---
  // In a real app, this runs in a WebWorker. Here we run it async on main thread
  // but "throttle" it to make it visual.
  
  const stepTraining = useCallback(async () => {
    if (!model) return;

    // Prevent making moves if the game is already over (waiting for reset)
    if (gameRef.current.isGameOver()) return;

    const turn = gameRef.current.turn();
    let selectedMove: Move | null = null;

    if (turn === 'w') {
      const mcts = await runMctsWithIndicator(gameRef.current, model, TRAINING_CONFIG, true);
      const policyVector = new Array(POLICY_OUTPUT_SIZE).fill(0);
      mcts.policy.forEach((entry) => {
        policyVector[moveToIndex(entry.move)] = entry.probability;
      });

      currentGameSamplesRef.current.push({
        fen: gameRef.current.fen(),
        policy: policyVector,
        player: 'w',
        value: 0
      });

      const movesWithProbs: MoveProbability[] = mcts.policy
        .map((entry) => ({
          san: entry.move.san,
          probability: entry.probability,
          isBest: false,
          move: entry.move
        }))
        .sort((a, b) => b.probability - a.probability);

      if (movesWithProbs.length > 0) movesWithProbs[0].isBest = true;
      setTopMoves(movesWithProbs.slice(0, 10));
      setHeatmap(
        movesWithProbs.map((mp) => ({
          square: mp.move.to,
          intensity: mp.probability
        }))
      );
      setCurrentMetrics((prev) => ({
        ...prev,
        entropy: computeEntropy(mcts.policy.map((entry) => entry.probability))
      }));

      const sampledIndex = sampleFromPolicy(mcts.policy.map((entry) => entry.probability));
      selectedMove = mcts.policy[sampledIndex]?.move ?? mcts.move;
    } else {
      const mcts = await runMctsWithIndicator(gameRef.current, model, MCTS_DIFFICULTY[difficulty]);
      selectedMove = mcts.move;
    }

    const applyMove = (move: Move): boolean => {
      try {
        const applied = gameRef.current.move({
          from: move.from,
          to: move.to,
          promotion: move.promotion
        });
        if (!applied) return false;
        setGame(new Chess(gameRef.current.fen())); // Update UI
        return true;
      } catch (e) {
        console.error("Invalid move attempted:", move, e);
        return false;
      }
    };

    if (selectedMove) {
      const applied = applyMove(selectedMove);
      if (!applied) {
        console.warn("Move rejected by engine:", selectedMove);
        const fallbackMoves = gameRef.current.moves({ verbose: true }) as Move[];
        if (fallbackMoves.length > 0) {
          const fallback = fallbackMoves[Math.floor(Math.random() * fallbackMoves.length)];
          if (applyMove(fallback)) {
            console.warn("Applied fallback random move:", fallback);
          }
        }
      }
    }

    // 6. Check Game End
    if (gameRef.current.isGameOver()) {
      const gameCount = currentMetrics.gamesPlayed + 1;
      let outcome = 0;
      if (gameRef.current.isCheckmate()) {
        outcome = gameRef.current.turn() === 'w' ? -1 : 1;
      }

      const finalizedSamples = currentGameSamplesRef.current.map((sample) => ({
        ...sample,
        value: sample.player === 'w' ? outcome : -outcome
      }));
      replayBufferRef.current.push(...finalizedSamples);
      currentGameSamplesRef.current = [];
      if (replayBufferRef.current.length > MAX_REPLAY_BUFFER) {
        replayBufferRef.current = replayBufferRef.current.slice(-MAX_REPLAY_BUFFER);
      }

      let policyLoss = currentMetrics.policyLoss;
      let valueLoss = currentMetrics.valueLoss;

      if (replayBufferRef.current.length >= BATCH_SIZE) {
        const batch = [];
        for (let i = 0; i < BATCH_SIZE; i++) {
          const sampleIndex = Math.floor(Math.random() * replayBufferRef.current.length);
          batch.push(replayBufferRef.current[sampleIndex]);
        }

        const inputTensors = batch.map((sample) => boardToTensor(new Chess(sample.fen)));
        const stateTensor = tf.concat(inputTensors, 0);
        inputTensors.forEach((t) => t.dispose());
        const policyTensor = tf.tensor2d(batch.map((s) => s.policy), [batch.length, POLICY_OUTPUT_SIZE]);
        const valueTargets = batch.map((s) => s.value);
        const valueTensor = tf.tensor2d(valueTargets, [batch.length, 1]);

        const history = await model.fit(stateTensor, [policyTensor, valueTensor], {
          batchSize: Math.min(BATCH_SIZE, batch.length),
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

      const newWinRate =
        (currentMetrics.winRate * currentMetrics.gamesPlayed + (outcome === 1 ? 1 : 0)) / gameCount;

      const newMetrics = {
        epoch: currentMetrics.epoch + 1,
        gamesPlayed: gameCount,
        policyLoss,
        valueLoss,
        winRate: newWinRate,
        entropy: currentMetrics.entropy
      };

      setCurrentMetrics(newMetrics);
      setMetricsHistory((prev) => {
        const updated = [...prev, newMetrics];
        return updated.slice(-50);
      });

      setTimeout(() => {
        gameRef.current.reset();
        setGame(new Chess());
      }, 1000);
    }

  }, [model, currentMetrics, difficulty, runMctsWithIndicator]);


  // Loop Effect
  useEffect(() => {
    let timeoutId: number;

    if (isTraining) {
        const loop = async () => {
            try {
                await stepTraining();
            } catch (err) {
                console.error("Training Loop Error:", err);
            }

            // Network moves fast, MCTS moves fast in this simplified version
            // Add slight delay for visual pacing
            const thinkingTime = Math.floor(Math.random() * 500) + 100;
            timeoutId = window.setTimeout(loop, thinkingTime);
        };
        loop();
    } 
    
    return () => {
        clearTimeout(timeoutId);
    };
  }, [isTraining, stepTraining]);


  // UI Handlers
  const handleStartTraining = () => {
    if (modelStatus !== 'ready') return;
    setIsTraining(true);
  };
  const handleStopTraining = () => {
    setIsTraining(false);
    setIsMctsThinking(false);
  };
  const handleReset = () => {
      setIsTraining(false);
      setIsMctsThinking(false);
      gameRef.current.reset();
      setGame(new Chess());
      setMetricsHistory([]);
      setCurrentMetrics(INITIAL_METRICS);
      setHeatmap([]);
      setTopMoves([]);
  };

  return (
    <div className="min-h-screen bg-neuro-900 text-gray-200 font-sans selection:bg-neuro-accent selection:text-neuro-900 p-4 lg:p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-neuro-700 pb-4">
        <div className="flex items-center space-x-3">
            <div className="bg-neuro-accent p-2 rounded-lg text-neuro-900 shadow-[0_0_15px_rgba(0,240,255,0.5)]">
                <Brain size={24} />
            </div>
            <div>
                <h1 className="text-2xl font-bold font-mono tracking-tighter text-white">
                    NEURO<span className="text-neuro-accent">CHESS</span>
                </h1>
                <p className="text-xs text-neuro-400 font-mono">TF.JS / RESNET / RL-ZERO</p>
            </div>
        </div>

        <div className="flex items-center space-x-4">
             <div className="hidden md:flex flex-col items-end mr-4">
                <span className="text-xs text-gray-500 font-mono">DEVICE</span>
                <span className="text-xs font-bold text-neuro-success flex items-center gap-1">
                     <Cpu size={12} /> GPU (WEBGL)
                </span>
             </div>
             <div className="flex flex-col items-end">
                <span className="text-xs text-gray-500 font-mono">MCTS DIFFICULTY</span>
                <select
                  value={difficulty}
                  onChange={(event) => setDifficulty(event.target.value as MctsDifficulty)}
                  className="bg-neuro-800 border border-neuro-600 text-xs font-mono text-gray-200 rounded px-2 py-1"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
             </div>
             <button 
                onClick={isTraining ? handleStopTraining : handleStartTraining}
                disabled={!isTraining && modelStatus !== 'ready'}
                className={`flex items-center gap-2 px-6 py-2 rounded-full font-bold transition-all duration-300 ${
                    isTraining 
                    ? 'bg-neuro-danger/10 text-neuro-danger border border-neuro-danger hover:bg-neuro-danger hover:text-white' 
                    : 'bg-neuro-accent text-neuro-900 hover:shadow-[0_0_20px_rgba(0,240,255,0.6)]'
                }`}
             >
                {isTraining ? <><StopCircle size={18} /> STOP TRAINING</> : <><Play size={18} /> START SELF-PLAY</>}
             </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:h-[calc(100vh-140px)] h-auto">
        
        {/* Left Column: Board (Approx 50-60% width) */}
        <div className="lg:col-span-6 xl:col-span-5 flex flex-col">
            <div className="flex-1 min-h-[400px] relative">
                <HeatmapBoard 
                    game={game} 
                    heatmap={heatmap} 
                    isBot={isTraining}
                    statusText={
                      modelStatus === 'loading'
                        ? 'INITIALIZING MODEL...'
                        : modelStatus === 'error'
                          ? 'MODEL INIT FAILED'
                          : isTraining
                            ? 'SELF-PLAY RUNNING'
                            : 'IDLE'
                    }
                />
            </div>
            {/* Status Bar under board */}
            <div className="mt-4 flex justify-between items-center bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                <div className="flex items-center gap-6">
                    <span className={`w-2 h-2 rounded-full ${isTraining ? 'bg-neuro-success animate-pulse' : 'bg-gray-500'}`}></span>
                    <div className="flex flex-col">
                         <span className="text-xs font-mono text-gray-400">STATUS</span>
                         <span className="text-sm font-bold font-mono text-gray-200">
                            {isTraining ? (
                                <span className="flex items-center gap-2">
                                    <span>EVAL:</span>
                                    <span className="text-white flex items-center gap-1"><Circle size={8} fill="white" /> NET</span>
                                    <span className="text-gray-500">vs</span>
                                    <span className="text-black bg-gray-600 rounded px-1 flex items-center gap-1"><Circle size={8} fill="black" /> MCTS</span>
                                </span>
                            ) : 'IDLE'}
                        </span>
                        {modelStatus === 'error' && modelError ? (
                          <span className="text-xs font-mono text-neuro-danger">{modelError}</span>
                        ) : null}
                    </div>
                    <div className="flex flex-col">
                        <span className="text-xs font-mono text-gray-400">TURN</span>
                        <span className="text-sm font-bold font-mono flex items-center gap-2">
                            <span
                              className={`w-2 h-2 rounded-full border border-gray-500 ${
                                game.turn() === 'w' ? 'bg-white' : 'bg-gray-900'
                              }`}
                            ></span>
                            <span className={game.turn() === 'w' ? 'text-white' : 'text-gray-300'}>
                              {game.turn() === 'w' ? 'WHITE' : 'BLACK'}
                            </span>
                        </span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-xs font-mono text-gray-400">MCTS</span>
                        <span
                          className={`text-sm font-bold font-mono flex items-center gap-2 ${
                            isMctsThinking ? 'text-neuro-accent' : 'text-gray-500'
                          }`}
                        >
                            <span
                              className={`w-2 h-2 rounded-full ${
                                isMctsThinking ? 'bg-neuro-accent animate-pulse' : 'bg-gray-600'
                              }`}
                            ></span>
                            {isMctsThinking ? 'THINKING' : 'IDLE'}
                        </span>
                    </div>
                </div>
                <button onClick={handleReset} className="text-gray-500 hover:text-white transition">
                    <RefreshCw size={18} />
                </button>
            </div>
        </div>

        {/* Middle Column: Stats (Charts) */}
        <div className="lg:col-span-6 xl:col-span-4 flex flex-col space-y-4">
            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">GAMES PLAYED</div>
                    <div className="text-2xl font-bold font-mono">{currentMetrics.gamesPlayed}</div>
                </div>
                <div className="bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">WIN RATE (vs MCTS)</div>
                    <div className="text-2xl font-bold font-mono text-neuro-success">
                        {(currentMetrics.winRate * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">EPOCH</div>
                    <div className="text-2xl font-bold font-mono text-neuro-accent">{currentMetrics.epoch}</div>
                </div>
                <div className="bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">VALUE LOSS</div>
                    <div className="text-2xl font-bold font-mono text-neuro-danger">
                        {currentMetrics.valueLoss.toFixed(4)}
                    </div>
                </div>
            </div>

            {/* Charts */}
            <LossChart data={metricsHistory} />
            <EntropyChart data={metricsHistory} />
        </div>

        {/* Right Column: Analysis (Moves) */}
        <div className="lg:col-span-12 xl:col-span-3 h-full flex flex-col min-h-0">
            <MoveAnalysis moves={topMoves} />
            
            <div className="mt-4 bg-neuro-800/50 p-4 rounded-lg border border-neuro-700 border-dashed shrink-0">
                <div className="flex items-center gap-2 mb-2">
                    <Activity size={16} className="text-neuro-400" />
                    <h4 className="text-sm font-bold text-gray-300">Architecture Info</h4>
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Model</span>
                        <span className="text-neuro-accent font-mono">TinyZero (ResNet)</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Blocks</span>
                        <span className="text-gray-300 font-mono">4 Residual</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Filters</span>
                        <span className="text-gray-300 font-mono">64</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Params</span>
                        <span className="text-gray-300 font-mono">~250k</span>
                    </div>
                     <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Inputs</span>
                        <span className="text-gray-300 font-mono">[8, 8, 18]</span>
                    </div>
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default App;
