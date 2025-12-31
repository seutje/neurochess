import React, { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { COLORS } from '../constants';
import { MoveProbability } from '../types';

interface Props {
  moves: MoveProbability[];
}

export const MctsVisualization: React.FC<Props> = React.memo(({ moves }) => {
  const chartData = useMemo(
    () =>
      moves.map((move, index) => {
        const label = move.san || `${move.move.from}${move.move.to}${move.move.promotion ?? ''}`;
        const visitCount = move.visitCount ?? Math.round(move.probability * 1000);
        return {
          index: index + 1,
          label,
          visitCount,
          probability: move.probability * 100
        };
      }),
    [moves]
  );

  const totalVisits = useMemo(
    () => chartData.reduce((acc, item) => acc + item.visitCount, 0),
    [chartData]
  );
  const bestMove = useMemo(() => moves.find((move) => move.isBest) ?? moves[0], [moves]);

  return (
    <div className="bg-neuro-800 p-3 border border-neuro-600 flex flex-col min-h-[320px]">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-mono text-gray-400 uppercase tracking-wider">MCTS Search Visualization</h3>
        {bestMove ? (
          <div className="text-xs font-mono text-gray-400">
            <span className="text-neuro-success">BEST</span> {bestMove.san || `${bestMove.move.from}${bestMove.move.to}`}
          </div>
        ) : null}
      </div>
      <div className="flex items-center gap-4 text-[11px] text-gray-400 font-mono mb-2">
        <span>
          VISITS <span className="text-gray-200">{totalVisits}</span>
        </span>
        <span>
          MOVES <span className="text-gray-200">{chartData.length}</span>
        </span>
      </div>
      <div className="flex-1 min-h-0">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-gray-500 font-mono italic">
            Awaiting MCTS data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} opacity={0.3} />
              <XAxis dataKey="index" stroke={COLORS.text} fontSize={10} tickLine={false} />
              <YAxis
                yAxisId="visits"
                stroke={COLORS.text}
                fontSize={10}
                width={36}
                tickLine={false}
              />
              <YAxis
                yAxisId="probability"
                orientation="right"
                stroke={COLORS.text}
                fontSize={10}
                width={36}
                domain={[0, 100]}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #3a3a3a' }}
                itemStyle={{ fontSize: '12px' }}
                labelFormatter={(label, payload) => {
                  const item = payload?.[0]?.payload;
                  return item?.label ? `${item.label}` : `Move ${label}`;
                }}
                formatter={(value, name) => {
                  if (name === 'probability') return [`${Number(value).toFixed(1)}%`, 'Probability'];
                  return [value, 'Visits'];
                }}
              />
              <Bar
                yAxisId="visits"
                dataKey="visitCount"
                fill={COLORS.policy}
                barSize={20}
                isAnimationActive={false}
              />
              <Line
                yAxisId="probability"
                type="monotone"
                dataKey="probability"
                stroke={COLORS.entropy}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
});
