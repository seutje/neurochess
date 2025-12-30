import React from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import { HeatmapSquare } from '../types';

interface Props {
  game: Chess;
  heatmap: HeatmapSquare[];
  onPieceDrop?: (source: string, target: string, piece: string) => boolean;
  isBot?: boolean;
}

export const HeatmapBoard: React.FC<Props> = ({ game, heatmap, onPieceDrop, isBot }) => {
  // Convert custom heatmap array to react-chessboard customSquareStyles
  const customSquareStyles = React.useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    
    // Base styles for all squares (optional, for aesthetics)
    
    // Apply heatmap
    heatmap.forEach(({ square, intensity }) => {
        // Red for low probability/danger, Green for high probability/best move
        // But typically Policy is just "probability of picking this move". 
        // Let's use Neuro-Accent color (Cyan) opacity.
        styles[square] = {
            backgroundColor: `rgba(0, 240, 255, ${intensity * 0.6})`,
            boxShadow: `inset 0 0 10px rgba(0, 240, 255, ${intensity * 0.8})`
        };
    });

    // Highlight last move
    const history = game.history({ verbose: true });
    if (history.length > 0) {
        const lastMove = history[history.length - 1];
        const source = lastMove.from;
        const target = lastMove.to;
        styles[source] = { backgroundColor: 'rgba(255, 255, 0, 0.2)' };
        styles[target] = { backgroundColor: 'rgba(255, 255, 0, 0.4)' };
    }

    return styles;
  }, [heatmap, game]);

  // Cast Chessboard to any to avoid type definition issues with 'position' prop in some versions
  const ChessboardComponent = Chessboard as any;

  return (
    <div className="w-full h-full relative group">
      <div className={`absolute -inset-1 bg-gradient-to-r from-neuro-accent to-purple-600 rounded-lg blur opacity-20 group-hover:opacity-40 transition duration-1000 ${isBot ? 'animate-pulse' : ''}`}></div>
      <div className="relative bg-neuro-900 rounded-lg p-1 border border-neuro-600 shadow-2xl">
        <ChessboardComponent 
            position={game.fen()} 
            onPieceDrop={onPieceDrop}
            customSquareStyles={customSquareStyles}
            customDarkSquareStyle={{ backgroundColor: '#2a2a40' }}
            customLightSquareStyle={{ backgroundColor: '#3e3e5e' }}
            arePiecesDraggable={!isBot}
        />
      </div>
    </div>
  );
};