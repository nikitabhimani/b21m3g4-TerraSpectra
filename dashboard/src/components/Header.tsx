import React from 'react';
import { 
  Activity, 
  Download, 
  Layers, 
  PlayCircle, 
  Radio, 
  Satellite, 
  Sparkles,
  Wifi,
  WifiOff 
} from 'lucide-react';

interface HeaderProps {
  apiStatus: string;
  isDemoMode: boolean;
  onToggleDemoMode: () => void;
  onOpenJobModal: () => void;
  onOpenExportModal: () => void;
  activeJobId: string;
}

export const Header: React.FC<HeaderProps> = ({
  apiStatus,
  isDemoMode,
  onToggleDemoMode,
  onOpenJobModal,
  onOpenExportModal,
  activeJobId,
}) => {
  const isOnline = !isDemoMode && (apiStatus === 'ok' || apiStatus === 'degraded' || apiStatus === 'healthy');

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-6 flex items-center justify-between z-30 select-none">
      {/* Brand & Project Identity */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 ring-1 ring-emerald-400/30">
          <Satellite className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
              TerraSpectra
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                GIS 3D
              </span>
            </h1>
          </div>
          <p className="text-xs text-slate-400 font-medium hidden sm:block">
            Hyperspectral Pre-Visual Crop Disease Forecast &bull; 3D-CNN + ViT Hybrid
          </p>
        </div>
      </div>

      {/* Center Status / Rehearsal Badges */}
      <div className="hidden lg:flex items-center gap-3 bg-slate-950/70 border border-slate-800/80 px-3 py-1.5 rounded-xl">
        <div className="flex items-center gap-2">
          {isOnline ? (
            <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <Wifi className="w-3.5 h-3.5" />
              API Live (Port 8000)
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs font-medium text-amber-400">
              <Radio className="w-3.5 h-3.5 animate-pulse" />
              Demo Fixture Active (Contract C4)
            </span>
          )}
        </div>
        <div className="h-3 w-px bg-slate-800" />
        <span className="text-xs text-slate-400 font-mono">
          Job: <span className="text-slate-200">{activeJobId.substring(0, 14)}...</span>
        </span>
      </div>

      {/* Right Controls & CTAs */}
      <div className="flex items-center gap-2.5">
        <button
          onClick={onToggleDemoMode}
          title="Toggle between live API and offline contract fixtures"
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 border ${
            isDemoMode
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20'
              : 'bg-slate-800/80 border-slate-700/80 text-slate-300 hover:bg-slate-700'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>{isDemoMode ? 'Demo Mode' : 'Live API Mode'}</span>
        </button>

        <button
          onClick={onOpenJobModal}
          className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-600/25 flex items-center gap-1.5 transition-all active:scale-95"
        >
          <PlayCircle className="w-3.5 h-3.5" />
          <span>Run Pre-Visual Scan</span>
        </button>

        <button
          onClick={onOpenExportModal}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 flex items-center gap-1.5 transition-all"
        >
          <Download className="w-3.5 h-3.5 text-slate-400" />
          <span className="hidden sm:inline">Export Plan</span>
        </button>
      </div>
    </header>
  );
};
