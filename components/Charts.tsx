import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrainingMetrics } from '../types';
import { COLORS } from '../constants';

interface ChartProps {
  data: TrainingMetrics[];
}

export const LossChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-48 w-full bg-neuro-800 rounded-lg p-2 border border-neuro-600">
      <h3 className="text-xs font-mono text-gray-400 mb-2 uppercase tracking-wider">Loss History</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} opacity={0.3} />
          <XAxis dataKey="epoch" hide />
          <YAxis stroke={COLORS.text} fontSize={10} width={30} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1c1c2e', border: '1px solid #3e3e5e' }}
            itemStyle={{ fontSize: '12px' }}
          />
          <Line 
            type="monotone" 
            dataKey="policyLoss" 
            stroke={COLORS.policy} 
            strokeWidth={2} 
            dot={false}
            animationDuration={300}
          />
          <Line 
            type="monotone" 
            dataKey="valueLoss" 
            stroke={COLORS.value} 
            strokeWidth={2} 
            dot={false}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const EntropyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-32 w-full bg-neuro-800 rounded-lg p-2 border border-neuro-600 mt-2">
      <h3 className="text-xs font-mono text-gray-400 mb-2 uppercase tracking-wider">Network Entropy (Uncertainty)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} opacity={0.3} />
          <XAxis dataKey="epoch" hide />
          <YAxis stroke={COLORS.text} fontSize={10} width={30} domain={[0, 'auto']} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1c1c2e', border: '1px solid #3e3e5e' }}
            itemStyle={{ fontSize: '12px' }}
          />
          <Area 
            type="monotone" 
            dataKey="entropy" 
            stroke={COLORS.entropy} 
            fill={COLORS.entropy} 
            fillOpacity={0.1}
            strokeWidth={2}
            animationDuration={300}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};