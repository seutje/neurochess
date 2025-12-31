import React, { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { Activity, Brain, Cpu, Play, StopCircle, RefreshCw, Circle, Flag } from 'lucide-react';

import { HeatmapBoard } from './components/HeatmapBoard';
import { MctsVisualization } from './components/MctsVisualization';
import { MoveAnalysis } from './components/MoveAnalysis';
import { TrainingMetrics, MoveProbability, HeatmapSquare, MctsDifficulty, PerformanceStats } from './types';

const INITIAL_METRICS: TrainingMetrics = {
  epoch: 0,
  gamesPlayed: 0,
  policyLoss: 2.5,
  valueLoss: 1.0,
  winRate: 0,
  entropy: 4.5
};

type WorkerStatus = 'loading' | 'ready' | 'error';

type WorkerStatusMessage = {
  type: 'status';
  status: WorkerStatus;
  error?: string;
  backend?: string;
  phase?: string;
};

type WorkerStateMessage = {
  type: 'state';
  fen: string;
  heatmap: HeatmapSquare[];
  topMoves: MoveProbability[];
  moveHistory: string[];
  currentMetrics: TrainingMetrics;
  isMctsThinking: boolean;
  perfStats: PerformanceStats;
};

type WorkerMessage = WorkerStatusMessage | WorkerStateMessage;

const App: React.FC = () => {
  // Game State
  const [game, setGame] = useState(new Chess());
  const [isTraining, setIsTraining] = useState(false);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState<string | null>(null);
  const [backend, setBackend] = useState<string>('unknown');
  const [modelPhase, setModelPhase] = useState<string>('initializing');
  const [difficulty, setDifficulty] = useState<MctsDifficulty>('easy');
  const [isMctsThinking, setIsMctsThinking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [winBanner, setWinBanner] = useState<string | null>(null);
  
  // Metrics & Visuals
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics>(INITIAL_METRICS);
  const [heatmap, setHeatmap] = useState<HeatmapSquare[]>([]);
  const [topMoves, setTopMoves] = useState<MoveProbability[]>([]);
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [perfStats, setPerfStats] = useState<PerformanceStats>({
    lastMctsMs: 0,
    avgMctsMs: 0,
    mctsRuns: 0,
    lastStepMs: 0,
    avgStepMs: 0,
    steps: 0,
    lastReadbackMs: 0,
    avgReadbackMs: 0,
    readbacks: 0
  });
  
  const workerRef = useRef<Worker | null>(null);
  const pauseTimeoutRef = useRef<number | null>(null);
  const pendingStateRef = useRef<WorkerStateMessage | null>(null);
  const pausedRef = useRef(false);

  useEffect(() => {
    const worker = new Worker(new URL('./workers/trainingWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === 'status') {
        setModelStatus(message.status);
        setModelError(message.status === 'error' ? message.error ?? 'Unknown error' : null);
        if (message.backend) setBackend(message.backend);
        if (message.phase) setModelPhase(message.phase);
        return;
      }
      if (pausedRef.current) {
        pendingStateRef.current = message;
        return;
      }
      const nextGame = new Chess(message.fen);
      setGame(nextGame);
      setHeatmap(message.heatmap);
      setTopMoves(message.topMoves);
      setMoveHistory(message.moveHistory);
      setCurrentMetrics(message.currentMetrics);
      setIsMctsThinking(message.isMctsThinking);
      setPerfStats(message.perfStats);

      if (nextGame.isCheckmate()) {
        const winner = nextGame.turn() === 'w' ? 'BLACK' : 'WHITE';
        setWinBanner(`${winner} WINS`);
        setIsPaused(true);
        pausedRef.current = true;
        if (pauseTimeoutRef.current) {
          window.clearTimeout(pauseTimeoutRef.current);
        }
        pauseTimeoutRef.current = window.setTimeout(() => {
          setIsPaused(false);
          setWinBanner(null);
          pausedRef.current = false;
          const pending = pendingStateRef.current;
          pendingStateRef.current = null;
          if (pending) {
            const resumedGame = new Chess(pending.fen);
            setGame(resumedGame);
            setHeatmap(pending.heatmap);
            setTopMoves(pending.topMoves);
            setMoveHistory(pending.moveHistory);
            setCurrentMetrics(pending.currentMetrics);
            setIsMctsThinking(pending.isMctsThinking);
            setPerfStats(pending.perfStats);
          }
        }, 3000);
      }
    };

    const baseRoot = new URL(import.meta.env.BASE_URL ?? '/', window.location.origin);
    const baseModelUrl = new URL('models/base/model.json', baseRoot).toString();
    worker.postMessage({ type: 'init', baseModelUrl });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (pauseTimeoutRef.current) {
        window.clearTimeout(pauseTimeoutRef.current);
      }
    };
  }, []);

  const isGpuBackend = backend === 'webgl' || backend === 'webgpu';
  const deviceLabel = (() => {
    switch (backend) {
      case 'webgl':
        return 'GPU (WEBGL)';
      case 'webgpu':
        return 'GPU (WEBGPU)';
      case 'wasm':
        return 'CPU (WASM)';
      case 'cpu':
        return 'CPU';
      default:
        return backend.toUpperCase();
    }
  })();
  const checkStatus = game.isCheckmate() ? 'CHECKMATE' : game.isCheck() ? 'CHECK' : null;
  const checkStatusText = checkStatus ? `${game.turn() === 'w' ? 'WHITE' : 'BLACK'} ${checkStatus}` : null;
  const canForfeit = !isTraining && !isPaused && modelStatus === 'ready';
  const loadingLabel =
    modelPhase === 'downloading'
      ? 'DOWNLOADING MODEL'
      : modelPhase === 'warming'
        ? 'WARMING UP'
        : modelPhase === 'building'
          ? 'BUILDING MODEL'
          : 'INITIALIZING';

  // UI Handlers
  const handleStartTraining = () => {
    if (modelStatus !== 'ready') return;
    handleReset();
    setIsTraining(true);
    workerRef.current?.postMessage({ type: 'start' });
  };
  const handleStopTraining = () => {
    setIsTraining(false);
    setIsMctsThinking(false);
    workerRef.current?.postMessage({ type: 'stop' });
  };
  const handleReset = () => {
      setIsTraining(false);
      setIsMctsThinking(false);
      setIsPaused(false);
      setWinBanner(null);
      pausedRef.current = false;
      pendingStateRef.current = null;
      if (pauseTimeoutRef.current) {
        window.clearTimeout(pauseTimeoutRef.current);
        pauseTimeoutRef.current = null;
      }
      setGame(new Chess());
      setCurrentMetrics(INITIAL_METRICS);
      setHeatmap([]);
      setTopMoves([]);
      setMoveHistory([]);
      workerRef.current?.postMessage({ type: 'reset' });
  };

  const handleForfeit = () => {
      const winner = 'BLACK';
      setWinBanner(`${winner} WINS`);
      setIsPaused(true);
      pausedRef.current = true;
      if (pauseTimeoutRef.current) {
        window.clearTimeout(pauseTimeoutRef.current);
        pauseTimeoutRef.current = null;
      }
      pauseTimeoutRef.current = window.setTimeout(() => {
        handleReset();
      }, 1200);
  };

  const handlePieceDrop = ({
    sourceSquare,
    targetSquare,
    pieceType
  }: {
    sourceSquare: string;
    targetSquare: string | null;
    pieceType: string;
  }) => {
    if (isTraining || isPaused || modelStatus !== 'ready') return false;
    if (game.turn() !== 'w') return false;
    if (!targetSquare) return false;
    const nextGame = new Chess(game.fen());
    const isPawn = pieceType.toLowerCase().endsWith('p');
    const isPromotionRank = targetSquare.endsWith('8') || targetSquare.endsWith('1');
    const promotion = isPawn && isPromotionRank ? 'q' : undefined;
    const result = nextGame.move({ from: sourceSquare, to: targetSquare, promotion });
    if (!result) return false;
    setGame(nextGame);
    setMoveHistory((prev) => [...prev, result.san]);
    workerRef.current?.postMessage({
      type: 'playerMove',
      move: { from: sourceSquare, to: targetSquare, promotion }
    });
    return true;
  };

  return (
    <div className="min-h-screen bg-neuro-900 text-gray-200 font-sans selection:bg-neuro-accent selection:text-neuro-900 p-4 lg:p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-neuro-700 pb-4">
        <div className="flex items-center space-x-3">
            <div className="bg-neuro-accent p-2 text-neuro-900 shadow-[0_0_15px_rgba(246,201,69,0.5)]">
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
                <span
                  className={`text-xs font-bold flex items-center gap-1 ${
                    isGpuBackend ? 'text-neuro-success' : 'text-gray-300'
                  }`}
                >
                     <Cpu size={12} /> {deviceLabel}
                </span>
             </div>
             <div className="flex flex-col items-end">
                <span className="text-xs text-gray-500 font-mono">MCTS DIFFICULTY</span>
                <select
                  value={difficulty}
                  onChange={(event) => {
                    const nextDifficulty = event.target.value as MctsDifficulty;
                    setDifficulty(nextDifficulty);
                    workerRef.current?.postMessage({ type: 'setDifficulty', difficulty: nextDifficulty });
                  }}
                  className="bg-neuro-800 border border-neuro-600 text-xs font-mono text-gray-200 px-2 py-1"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
             </div>
             <button 
                onClick={isTraining ? handleStopTraining : handleStartTraining}
                disabled={!isTraining && modelStatus !== 'ready'}
                className={`flex items-center gap-2 px-6 py-2 font-bold transition-all duration-300 ${
                    isTraining 
                    ? 'bg-neuro-danger/10 text-neuro-danger border border-neuro-danger hover:bg-neuro-danger hover:text-white' 
                    : 'bg-neuro-accent text-neuro-900 hover:shadow-[0_0_20px_rgba(246,201,69,0.6)]'
                }`}
             >
                {isTraining ? <><StopCircle size={18} /> STOP TRAINING</> : <><Play size={18} /> START SELF-PLAY</>}
             </button>
        </div>
      </header>

      {modelStatus === 'loading' ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-neuro-900/70 backdrop-blur-sm">
          <div className="w-[320px] max-w-[80vw] border border-neuro-700 bg-neuro-800/90 px-6 py-5 shadow-[0_0_24px_rgba(246,201,69,0.2)]">
            <div className="text-xs font-mono text-gray-400 mb-3 uppercase tracking-wider">
              Booting Neural Core
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono text-gray-500 mb-2">
              <span>STATUS</span>
              <span>{loadingLabel}</span>
            </div>
            <div className="h-2 bg-neuro-900 border border-neuro-700 overflow-hidden">
              <div className="h-full w-1/3 bg-neuro-accent animate-loading-bar"></div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:h-[calc(100vh-140px)] h-auto">
        
        {/* Left Column: Board (Approx 50-60% width) */}
        <div className="lg:col-span-6 xl:col-span-5 flex flex-col">
            <div className="flex-1 min-h-[400px] relative">
                <HeatmapBoard 
                    game={game} 
                    heatmap={heatmap} 
                    onPieceDrop={handlePieceDrop}
                    isBot={isTraining}
                    isPaused={isPaused}
                    alertText={checkStatusText ?? undefined}
                    overlayText={winBanner ?? undefined}
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
            <div className="mt-4 flex justify-between items-center bg-neuro-800 p-3 border border-neuro-700">
                <div className="flex items-center gap-6">
                    <span className={`w-2 h-2 ${isTraining ? 'bg-neuro-success animate-pulse' : 'bg-gray-500'}`}></span>
                    <div className="flex flex-col">
                         <span className="text-xs font-mono text-gray-400">STATUS</span>
                         <span className="text-sm font-bold font-mono text-gray-200">
                            {isTraining ? (
                                <span className="flex items-center gap-2">
                                    <span>EVAL:</span>
                                    <span className="text-white flex items-center gap-1"><Circle size={8} fill="white" /> NET</span>
                                    <span className="text-gray-500">vs</span>
                                    <span className="text-neuro-900 bg-neuro-400 px-1 flex items-center gap-1"><Circle size={8} fill="black" /> MCTS</span>
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
                              className={`w-2 h-2 border border-gray-500 ${
                                game.turn() === 'w' ? 'bg-white' : 'bg-gray-900'
                              }`}
                            ></span>
                            <span className={game.turn() === 'w' ? 'text-white' : 'text-gray-300'}>
                              {game.turn() === 'w' ? 'WHITE' : 'BLACK'}
                            </span>
                        </span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-xs font-mono text-gray-400">CHECK</span>
                        <span
                          className={`text-sm font-bold font-mono ${
                            checkStatus ? 'text-neuro-danger' : 'text-gray-500'
                          }`}
                        >
                          {checkStatusText ?? '—'}
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
                              className={`w-2 h-2 ${
                                isMctsThinking ? 'bg-neuro-accent animate-pulse' : 'bg-gray-600'
                              }`}
                            ></span>
                            {isMctsThinking ? 'THINKING' : 'IDLE'}
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <button
                      onClick={handleForfeit}
                      disabled={!canForfeit}
                      className={`transition ${
                        canForfeit ? 'text-gray-500 hover:text-neuro-danger' : 'text-gray-700 cursor-not-allowed'
                      }`}
                      aria-label="Forfeit game"
                    >
                      <Flag size={18} />
                    </button>
                    <button onClick={handleReset} className="text-gray-500 hover:text-white transition">
                        <RefreshCw size={18} />
                    </button>
                </div>
            </div>
        </div>

        {/* Middle Column: Stats (Charts) */}
        <div className="lg:col-span-6 xl:col-span-4 flex flex-col space-y-4">
            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-neuro-800 p-3 border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">GAMES PLAYED</div>
                    <div className="text-2xl font-bold font-mono">{currentMetrics.gamesPlayed}</div>
                </div>
                <div className="bg-neuro-800 p-3 border border-neuro-700">
                    <div className="text-xs text-neuro-400 font-mono mb-1">WIN RATE (vs heuristic)</div>
                    <div className="text-2xl font-bold font-mono text-neuro-success">
                        {(currentMetrics.winRate * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            {/* MCTS Visualization */}
            <MctsVisualization moves={topMoves} />
            <div className="bg-neuro-800 p-3 border border-neuro-600 flex flex-col min-h-[160px] max-h-56">
                <h3 className="text-xs font-mono text-gray-400 mb-2 uppercase tracking-wider">Moves Log</h3>
                <div className="flex-1 overflow-y-auto pr-1 space-y-1">
                  {moveHistory.length === 0 ? (
                    <p className="text-xs text-gray-500 font-mono italic">Awaiting first move...</p>
                  ) : (
                    (() => {
                      const totalPairs = Math.ceil(moveHistory.length / 2);
                      return Array.from({ length: totalPairs }).map((_, idx) => {
                        const pairIndex = totalPairs - 1 - idx;
                      const whiteMove = moveHistory[pairIndex * 2];
                      const blackMove = moveHistory[pairIndex * 2 + 1];
                      return (
                        <div key={`move-${idx}`} className="grid grid-cols-[32px_1fr_1fr] gap-2 text-xs font-mono">
                          <span className="text-gray-500">{pairIndex + 1}.</span>
                          <span className="text-gray-200">{whiteMove ?? ''}</span>
                          <span className="text-gray-400">{blackMove ?? ''}</span>
                        </div>
                      );
                      });
                    })()
                  )}
                </div>
            </div>
        </div>

        {/* Right Column: Analysis (Moves) */}
        <div className="lg:col-span-12 xl:col-span-3 h-full flex flex-col min-h-0">
            <MoveAnalysis moves={topMoves} />
            
            <div className="mt-4 bg-neuro-800/70 p-4 border border-neuro-700 shrink-0">
                <div className="flex items-center gap-2 mb-2">
                    <Activity size={16} className="text-neuro-400" />
                    <h4 className="text-sm font-bold text-gray-300">Performance</h4>
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">MCTS (last)</span>
                        <span className="text-gray-300 font-mono">{perfStats.lastMctsMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">MCTS (avg)</span>
                        <span className="text-gray-300 font-mono">{perfStats.avgMctsMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Step (last)</span>
                        <span className="text-gray-300 font-mono">{perfStats.lastStepMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Step (avg)</span>
                        <span className="text-gray-300 font-mono">{perfStats.avgStepMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Readback (last)</span>
                        <span className="text-gray-300 font-mono">{perfStats.lastReadbackMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Readback (avg)</span>
                        <span className="text-gray-300 font-mono">{perfStats.avgReadbackMs.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Runs</span>
                        <span className="text-gray-300 font-mono">
                          {perfStats.mctsRuns} MCTS / {perfStats.steps} steps / {perfStats.readbacks} readbacks
                        </span>
                    </div>
                </div>
            </div>

            <div className="mt-4 bg-neuro-800/50 p-4 border border-neuro-700 border-dashed shrink-0">
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
