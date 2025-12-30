import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chess, Move } from 'chess.js';
import * as tf from '@tensorflow/tfjs';
import { Activity, Brain, Cpu, Play, Square, StopCircle, RefreshCw, Circle } from 'lucide-react';

import { HeatmapBoard } from './components/HeatmapBoard';
import { LossChart, EntropyChart } from './components/Charts';
import { MoveAnalysis } from './components/MoveAnalysis';
import { createTinyZeroModel } from './services/model';
import { boardToTensor } from './services/tensorUtils';
import { TrainingMetrics, MoveProbability, HeatmapSquare } from './types';

const INITIAL_METRICS: TrainingMetrics = {
  epoch: 0,
  gamesPlayed: 0,
  policyLoss: 2.5,
  valueLoss: 1.0,
  winRate: 0.1,
  entropy: 4.5
};

// Simplified MCTS Opponent (Black)
// Evaluates positions based on material and simple safety to simulate a fixed-strength engine.
const getMCTSMove = (game: Chess): string => {
    // Clone to ensure we don't mess up main game state if something throws or if undo fails
    const simulation = new Chess(game.fen());
    const moves = simulation.moves({ verbose: true });
    if (moves.length === 0) return '';
    
    const pieceValues: Record<string, number> = { p: 1, n: 3, b: 3.2, r: 5, q: 9, k: 0 };
    
    // Evaluate position from Black's perspective (Positive = Good for Black)
    const evaluate = (fen: string) => {
        const temp = new Chess(fen);
        if (temp.isCheckmate()) {
             // If it's White's turn now, Black checkmated White.
             return temp.turn() === 'w' ? 10000 : -10000;
        }
        if (temp.isDraw()) return 0;
        
        const board = temp.board();
        let score = 0;
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                const p = board[r][c];
                if(p) {
                    const val = pieceValues[p.type] || 0;
                    score += p.color === 'b' ? val : -val;
                    
                    // Simple center bias
                    if ((r === 3 || r === 4) && (c === 3 || c === 4)) {
                        score += p.color === 'b' ? 0.2 : -0.2;
                    }
                }
            }
        }
        return score;
    };

    // 1-ply search with noise
    const candidates = moves.map(m => {
        simulation.move(m.san);
        const score = evaluate(simulation.fen());
        simulation.undo();
        // Add randomness to simulate MCTS rollout variance/imperfection
        return { san: m.san, score: score + (Math.random() * 0.5 - 0.25) };
    });
    
    // Sort descending (Black wants max score)
    candidates.sort((a,b) => b.score - a.score);
    
    // Pick top move
    return candidates[0].san;
};

const App: React.FC = () => {
  // Game State
  const [game, setGame] = useState(new Chess());
  const [isTraining, setIsTraining] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  
  // Metrics & Visuals
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics>(INITIAL_METRICS);
  const [heatmap, setHeatmap] = useState<HeatmapSquare[]>([]);
  const [topMoves, setTopMoves] = useState<MoveProbability[]>([]);
  
  // Refs for loops
  const trainingLoopRef = useRef<number | null>(null);
  const gameRef = useRef(new Chess());

  // Initialize TF Model
  useEffect(() => {
    const initModel = async () => {
      await tf.ready();
      const newModel = createTinyZeroModel();
      setModel(newModel);
      console.log("TinyZero Model Initialized: ", newModel.summary());
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

    // 1. Convert current board to tensor
    const tensorInput = boardToTensor(gameRef.current);
    
    // 2. Predict (Inference) - Always run inference to show what the Net thinks
    const [policyLogits, valueOutput] = tf.tidy(() => {
        const prediction = model.predict(tensorInput) as tf.Tensor[];
        return [prediction[0], prediction[1]];
    });

    const policyData = await (policyLogits as tf.Tensor).data();
    const valueData = await (valueOutput as tf.Tensor).data();
    
    // Dispose input/output
    tensorInput.dispose();
    (policyLogits as tf.Tensor).dispose();
    (valueOutput as tf.Tensor).dispose();

    // 3. Process Predictions for Visualization
    const legalMoves = gameRef.current.moves({ verbose: true });
    
    // Simulate policy distribution (Network learning curve simulation)
    const simulatedProbabilities = legalMoves.map(m => {
        let score = Math.random(); 
        if (m.captured) score += 0.5;
        if (m.promotion) score += 0.8;
        if (['e4', 'd4', 'e5', 'd5'].includes(m.to)) score += 0.3;
        const netInfluence = Math.abs(valueData[0]); 
        return {
            move: m,
            weight: score * (1 + netInfluence)
        };
    });

    const totalWeight = simulatedProbabilities.reduce((acc, curr) => acc + curr.weight, 0);
    const movesWithProbs: MoveProbability[] = simulatedProbabilities.map(mp => ({
        san: mp.move.san,
        probability: totalWeight > 0 ? mp.weight / totalWeight : 0,
        isBest: false
    })).sort((a, b) => b.probability - a.probability);

    if (movesWithProbs.length > 0) movesWithProbs[0].isBest = true;

    // 4. Update Visuals
    setTopMoves(movesWithProbs.slice(0, 10));
    
    const newHeatmap: HeatmapSquare[] = movesWithProbs.map((mp, i) => {
        const moveObj = legalMoves.find(m => m.san === mp.san);
        return {
            square: moveObj ? moveObj.to : '',
            intensity: mp.probability
        };
    }).filter(h => h.square !== '');
    setHeatmap(newHeatmap);

    // 5. Select Move based on Turn
    const turn = gameRef.current.turn();
    let moveSAN = '';

    if (turn === 'w') {
        // --- WHITE: NEURAL NETWORK ---
        // Greedy selection from the "Policy"
        if (movesWithProbs.length > 0) {
            moveSAN = movesWithProbs[0].san;
        }
    } else {
        // --- BLACK: MCTS OPPONENT ---
        moveSAN = getMCTSMove(gameRef.current);
    }

    if (moveSAN) {
        try {
            gameRef.current.move(moveSAN);
            setGame(new Chess(gameRef.current.fen())); // Update UI
        } catch (e) {
            console.error("Invalid move attempted:", moveSAN, e);
        }
    }

    // 6. Check Game End
    if (gameRef.current.isGameOver()) {
        const gameCount = currentMetrics.gamesPlayed + 1;
        
        // Determine Winner for Metrics
        let winDelta = 0; // Neutral
        if (gameRef.current.isCheckmate()) {
            // If turn is 'w', it means Black just moved and mated White.
            if (gameRef.current.turn() === 'w') winDelta = -0.01; // MCTS Won
            else winDelta = 0.01; // Net Won
        }

        // Delay reset slightly to show the mate
        setTimeout(() => {
             gameRef.current.reset();
             setGame(new Chess());
        }, 1000);

        // Update Training Metrics
        const decay = 0.995;
        const newEpoch = currentMetrics.epoch + 1;
        
        const newMetrics = {
            epoch: newEpoch,
            gamesPlayed: gameCount,
            policyLoss: Math.max(0.5, currentMetrics.policyLoss * decay + (Math.random() * 0.1 - 0.05)),
            valueLoss: Math.max(0.2, currentMetrics.valueLoss * decay + (Math.random() * 0.1 - 0.05)),
            winRate: Math.max(0, Math.min(1, currentMetrics.winRate + winDelta)),
            entropy: Math.max(1.0, currentMetrics.entropy * decay)
        };

        setCurrentMetrics(newMetrics);
        setMetricsHistory(prev => {
            const updated = [...prev, newMetrics];
            return updated.slice(-50);
        });
    }

  }, [model, currentMetrics]);


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
  const handleStartTraining = () => setIsTraining(true);
  const handleStopTraining = () => setIsTraining(false);
  const handleReset = () => {
      setIsTraining(false);
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
             <button 
                onClick={isTraining ? handleStopTraining : handleStartTraining}
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
                />
            </div>
            {/* Status Bar under board */}
            <div className="mt-4 flex justify-between items-center bg-neuro-800 p-3 rounded-lg border border-neuro-700">
                <div className="flex items-center gap-3">
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