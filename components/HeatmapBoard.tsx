import React from 'react';
import { Chessboard, fenStringToPositionObject } from 'react-chessboard';
import { Chess } from 'chess.js';
import { HeatmapSquare } from '../types';

interface Props {
  game: Chess;
  heatmap: HeatmapSquare[];
  onPieceDrop?: (source: string, target: string, piece: string) => boolean;
  isBot?: boolean;
  statusText?: string;
  alertText?: string;
}

export const HeatmapBoard: React.FC<Props> = ({ game, heatmap, onPieceDrop, isBot, statusText, alertText }) => {
  const fen = game.fen();
  const piecePlacement = fen.split(' ')[0];
  const position = React.useMemo(() => fenStringToPositionObject(piecePlacement, 8, 8), [piecePlacement]);
  // Convert custom heatmap array to react-chessboard customSquareStyles
  const customSquareStyles = React.useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};

    const mergeSquareStyle = (square: string, next: React.CSSProperties) => {
      styles[square] = { ...(styles[square] ?? {}), ...next };
    };

    // Apply heatmap
    heatmap.forEach(({ square, intensity }) => {
      // Red for low probability/danger, Green for high probability/best move
      // But typically Policy is just "probability of picking this move".
      // Let's use Neuro-Accent color (yellow) opacity.
      mergeSquareStyle(square, {
        backgroundColor: `rgba(246, 201, 69, ${intensity * 0.6})`,
        boxShadow: `inset 0 0 10px rgba(246, 201, 69, ${intensity * 0.8})`
      });
    });

    // Highlight last move + animate
    const history = game.history({ verbose: true });
    if (history.length > 0) {
      const lastMove = history[history.length - 1];
      const source = lastMove.from;
      const target = lastMove.to;
      mergeSquareStyle(source, {
        backgroundColor: 'rgba(255, 255, 0, 0.2)',
        animation: 'movePulse 700ms ease-out'
      });
      mergeSquareStyle(target, {
        backgroundColor: 'rgba(255, 255, 0, 0.45)',
        boxShadow: 'inset 0 0 18px rgba(255, 255, 0, 0.4)',
        animation: 'movePulseStrong 900ms ease-out'
      });
    }

    return styles;
  }, [heatmap, game]);

  // Cast Chessboard to any to avoid type definition issues with 'position' prop in some versions
  const ChessboardComponent = Chessboard as any;

  return (
    <div className="w-full h-full relative group">
      <div className={`absolute -inset-1 bg-gradient-to-r from-neuro-accent to-neuro-400 blur opacity-20 group-hover:opacity-40 transition duration-1000 ${isBot ? 'animate-pulse' : ''}`}></div>
      <div className="relative bg-neuro-900 p-1 border border-neuro-600 shadow-2xl">
        <ChessboardComponent 
            options={{
              position,
              showAnimations: true,
              animationDurationInMs: 260,
              onPieceDrop,
              squareStyles: customSquareStyles,
              darkSquareStyle: { backgroundColor: '#1d1d1d' },
              lightSquareStyle: { backgroundColor: '#2a2a2a' },
              allowDragging: !isBot
            }}
        />
        {statusText ? (
          <div className="absolute top-2 left-2 bg-neuro-900/70 text-neuro-200 text-xs font-mono px-2 py-1 border border-neuro-700">
            {statusText}
          </div>
        ) : null}
        {alertText ? (
          <div
            className={`absolute top-2 right-2 text-xs font-mono px-2 py-1 border ${
              alertText.includes('CHECKMATE')
                ? 'bg-neuro-danger/20 text-neuro-danger border-neuro-danger'
                : 'bg-neuro-accent/10 text-neuro-accent border-neuro-accent'
            }`}
          >
            {alertText}
          </div>
        ) : null}
      </div>
    </div>
  );
};
