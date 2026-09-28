import React, { useEffect, useState } from 'react';
import { Calendar, FastForward, Pause, Play, RotateCcw } from 'lucide-react';

interface ForecastSliderProps {
  daysHorizon: number;
  onChangeDays: (days: number) => void;
}

export const ForecastSlider: React.FC<ForecastSliderProps> = ({
  daysHorizon,
  onChangeDays,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      timer = setInterval(() => {
        if (daysHorizon <= 1) {
          setIsPlaying(false);
          onChangeDays(0);
        } else {
          onChangeDays(daysHorizon - 1);
        }
      }, 700);
    }
    return () => clearInterval(timer);
  }, [isPlaying, daysHorizon, onChangeDays]);

  const handleReset = () => {
    setIsPlaying(false);
    onChangeDays(30);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 select-none backdrop-blur-md">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-semibold text-white">
            Forecast Horizon Timeline
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            T - {daysHorizon} Days
          </span>
          <span className="text-[11px] text-slate-400">
            {daysHorizon === 0
              ? 'Today (Visible Outbreak)'
              : daysHorizon <= 7
              ? 'Acute Blight Spread'
              : 'Pre-Visual Incubation'}
          </span>
        </div>
      </div>

      {/* Slider Control */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          title={isPlaying ? 'Pause timeline animation' : 'Play timeline progression'}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5 text-amber-400" /> : <Play className="w-3.5 h-3.5 text-emerald-400" />}
        </button>

        <button
          onClick={handleReset}
          title="Reset to 30 days prior"
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
        </button>

        <div className="relative flex-1 flex items-center">
          <input
            type="range"
            min="0"
            max="30"
            step="1"
            value={daysHorizon}
            onChange={(e) => onChangeDays(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
        </div>

        <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
          <span>0d</span>
          <span>&bull;</span>
          <span>30d</span>
        </div>
      </div>
    </div>
  );
};
