import React from 'react';
import type { ModelLayerSummary, TensorSummary } from '../types';
import { COLORS } from '../constants';

type Props = {
  layers: ModelLayerSummary[];
};

const formatNumber = (value: number) => value.toFixed(4);

const getRangeMetrics = (summary: TensorSummary) => {
  const maxAbs = Math.max(Math.abs(summary.min), Math.abs(summary.max), 1e-6);
  const range = maxAbs * 2;
  const minPct = ((summary.min + maxAbs) / range) * 100;
  const maxPct = ((summary.max + maxAbs) / range) * 100;
  const meanPct = ((summary.mean + maxAbs) / range) * 100;
  return { minPct, maxPct, meanPct };
};

const SummaryBar: React.FC<{
  summary: TensorSummary;
  color: string;
  label: string;
}> = ({ summary, color, label }) => {
  const { minPct, maxPct, meanPct } = getRangeMetrics(summary);
  const barWidth = Math.max(2, maxPct - minPct);

  return (
    <div className="flex items-center gap-3">
      <div className="w-10 text-[10px] font-mono text-gray-500 uppercase">{label}</div>
      <div className="relative flex-1 h-2 bg-neuro-900/60">
        <div className="absolute inset-y-0 left-1/2 w-px bg-neuro-700/70" />
        <div
          className="absolute h-2"
          style={{
            left: `${minPct}%`,
            width: `${barWidth}%`,
            backgroundColor: color
          }}
        />
        <div
          className="absolute -top-0.5 w-1.5 h-1.5 border border-neuro-900"
          style={{
            left: `${meanPct}%`,
            backgroundColor: '#ffffff'
          }}
        />
      </div>
      <div className="w-32 text-[10px] font-mono text-gray-500 text-right">
        mu {formatNumber(summary.mean)} sd {formatNumber(summary.std)}
      </div>
    </div>
  );
};

export const ModelParams: React.FC<Props> = ({ layers }) => {
  if (!layers.length) {
    return (
      <div className="bg-neuro-800/60 p-4 border border-neuro-700">
        <h4 className="text-sm font-bold text-gray-300 mb-2">Weights & Biases</h4>
        <div className="text-xs font-mono text-gray-500">Waiting for model parameters...</div>
      </div>
    );
  }

  return (
    <div className="bg-neuro-800/60 p-4 border border-neuro-700 flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-bold text-gray-300">Weights & Biases</h4>
        <span className="text-[10px] font-mono text-gray-500">{layers.length} layers</span>
      </div>
      <div className="space-y-3 overflow-y-auto max-h-80 pr-1">
        {layers.map((layer) => (
          <div key={layer.name} className="space-y-2">
            <div className="text-xs font-mono text-neuro-400">{layer.name}</div>
            {layer.weight ? (
              <SummaryBar summary={layer.weight} color={COLORS.policy} label="W" />
            ) : null}
            {layer.bias ? (
              <SummaryBar summary={layer.bias} color={COLORS.value} label="B" />
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
};
