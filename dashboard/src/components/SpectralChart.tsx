import React, { useState } from 'react';
import { SAMPLE_SPECTRAL_PROFILE } from '../fixtures/mockData';
import { SpectralPoint } from '../types';
import { Activity, Info } from 'lucide-react';

interface SpectralChartProps {
  dominantIndicator?: string;
}

export const SpectralChart: React.FC<SpectralChartProps> = ({ dominantIndicator }) => {
  const [hoveredPoint, setHoveredPoint] = useState<SpectralPoint | null>(null);
  const data = SAMPLE_SPECTRAL_PROFILE;

  // SVG dimensions
  const width = 460;
  const height = 180;
  const padding = { top: 20, right: 20, bottom: 30, left: 40 };

  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const minWl = 400;
  const maxWl = 2500;
  const minVal = 0.0;
  const maxVal = 0.65;

  const getX = (wl: number) => padding.left + ((wl - minWl) / (maxWl - minWl)) * chartW;
  const getY = (val: number) => padding.top + chartH - ((val - minVal) / (maxVal - minVal)) * chartH;

  // Build SVG path strings
  const healthyPath = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(d.wavelength).toFixed(1)} ${getY(d.healthy).toFixed(1)}`)
    .join(' ');

  const stressedPath = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(d.wavelength).toFixed(1)} ${getY(d.stressed).toFixed(1)}`)
    .join(' ');

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 select-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          <h4 className="text-xs font-semibold text-white">200-Band Hyperspectral Profile (C1)</h4>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <span className="w-2.5 h-0.5 bg-emerald-400 rounded-full" />
            Healthy
          </span>
          <span className="flex items-center gap-1.5 text-amber-400 font-medium">
            <span className="w-2.5 h-0.5 bg-amber-400 rounded-full" />
            Stressed (Simulated PROSAIL)
          </span>
        </div>
      </div>

      {/* SVG Canvas Chart */}
      <div className="relative w-full overflow-hidden flex justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-h-[190px]"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {/* Diagnostic Band Callout Bands */}
          {/* PRI Band (531-570nm) */}
          <rect
            x={getX(520)}
            y={padding.top}
            width={getX(580) - getX(520)}
            height={chartH}
            fill="#f59e0b"
            opacity="0.12"
          />
          {/* Red-Edge Shift Band (690-740nm) */}
          <rect
            x={getX(680)}
            y={padding.top}
            width={getX(750) - getX(680)}
            height={chartH}
            fill="#6366f1"
            opacity="0.15"
          />
          {/* Water Absorption (970nm) */}
          <rect
            x={getX(940)}
            y={padding.top}
            width={getX(1010) - getX(940)}
            height={chartH}
            fill="#06b6d4"
            opacity="0.12"
          />

          {/* Grid lines */}
          {[0.1, 0.3, 0.5].map((val) => (
            <g key={val}>
              <line
                x1={padding.left}
                y1={getY(val)}
                x2={width - padding.right}
                y2={getY(val)}
                stroke="#1e293b"
                strokeDasharray="2 3"
              />
              <text
                x={padding.left - 6}
                y={getY(val) + 3}
                fill="#64748b"
                fontSize="9"
                textAnchor="end"
                fontFamily="monospace"
              >
                {val.toFixed(1)}
              </text>
            </g>
          ))}

          {/* X axis labels */}
          {[500, 750, 1000, 1500, 2000, 2500].map((wl) => (
            <text
              key={wl}
              x={getX(wl)}
              y={height - 10}
              fill="#64748b"
              fontSize="9"
              textAnchor="middle"
              fontFamily="monospace"
            >
              {wl}
            </text>
          ))}

          {/* Lines */}
          <path
            d={healthyPath}
            fill="none"
            stroke="#10b981"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <path
            d={stressedPath}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray="4 2"
          />

          {/* Interactive invisible hit targets */}
          {data.map((d) => (
            <circle
              key={d.wavelength}
              cx={getX(d.wavelength)}
              cy={getY(d.stressed)}
              r="6"
              fill="transparent"
              className="cursor-pointer hover:stroke-white hover:stroke-2"
              onMouseEnter={() => setHoveredPoint(d)}
            />
          ))}

          {/* Active Hover Marker */}
          {hoveredPoint && (
            <g>
              <line
                x1={getX(hoveredPoint.wavelength)}
                y1={padding.top}
                x2={getX(hoveredPoint.wavelength)}
                y2={padding.top + chartH}
                stroke="#94a3b8"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              <circle
                cx={getX(hoveredPoint.wavelength)}
                cy={getY(hoveredPoint.healthy)}
                r="4"
                fill="#10b981"
              />
              <circle
                cx={getX(hoveredPoint.wavelength)}
                cy={getY(hoveredPoint.stressed)}
                r="4"
                fill="#f59e0b"
              />
            </g>
          )}
        </svg>
      </div>

      {/* Hover Info / Diagnostic Tooltip */}
      <div className="mt-2.5 pt-2 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono">
        {hoveredPoint ? (
          <>
            <span className="text-slate-300 font-semibold">
              &lambda;: {hoveredPoint.wavelength} nm
            </span>
            <span className="text-emerald-400">
              Refl(H): {(hoveredPoint.healthy * 100).toFixed(1)}%
            </span>
            <span className="text-amber-400">
              Refl(S): {(hoveredPoint.stressed * 100).toFixed(1)}%
            </span>
            <span className="text-indigo-400">
              Attr: {hoveredPoint.importance > 0.008 ? 'Diagnostic Peak' : 'Low'}
            </span>
          </>
        ) : (
          <div className="flex items-center gap-1.5 text-slate-400 font-sans text-[11px]">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span>Hover across wavelengths to inspect red-edge blue shift & PRI drops.</span>
          </div>
        )}
      </div>
    </div>
  );
};
