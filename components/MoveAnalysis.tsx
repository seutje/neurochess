import React from 'react';
import { MoveProbability } from '../types';

interface Props {
  moves: MoveProbability[];
}

export const MoveAnalysis: React.FC<Props> = ({ moves }) => {
  return (
    <div className="bg-neuro-800 rounded-lg p-4 border border-neuro-600 flex-1 min-h-[320px] overflow-y-auto">
      <h3 className="text-xs font-mono text-neuro-accent mb-4 uppercase tracking-wider border-b border-neuro-600 pb-2">
        Policy Head Analysis
      </h3>
      <div className="space-y-2">
        {moves.length === 0 ? (
            <p className="text-xs text-gray-500 font-mono italic">Waiting for inference...</p>
        ) : (
            moves.map((move, idx) => (
            <div key={idx} className="flex items-center justify-between group">
                <div className="flex items-center space-x-3">
                <span className={`text-xs font-mono w-4 text-gray-500`}>{idx + 1}.</span>
                <span className={`font-bold font-mono ${move.isBest ? 'text-neuro-success' : 'text-gray-300'}`}>
                    {move.san}
                </span>
                </div>
                <div className="flex items-center space-x-2 flex-1 mx-3">
                <div className="h-1.5 flex-1 bg-neuro-900 rounded-full overflow-hidden">
                    <div 
                    className={`h-full rounded-full ${move.isBest ? 'bg-neuro-success' : 'bg-neuro-400'}`}
                    style={{ width: `${move.probability * 100}%` }}
                    />
                </div>
                </div>
                <span className="text-xs font-mono text-gray-400 w-12 text-right">
                {(move.probability * 100).toFixed(1)}%
                </span>
            </div>
            ))
        )}
      </div>
    </div>
  );
};