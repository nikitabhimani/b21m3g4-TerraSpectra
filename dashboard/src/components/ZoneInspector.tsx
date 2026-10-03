import React from 'react';
import { 
  AlertCircle, 
  Calendar, 
  CheckCircle2, 
  ChevronRight, 
  Clock, 
  Compass, 
  Crosshair, 
  Flame, 
  Leaf, 
  ShieldAlert, 
  Sparkles, 
  Wheat 
} from 'lucide-react';
import { ZoneFeature, ZoneProperties } from '../types';

interface ZoneInspectorProps {
  selectedZone: ZoneFeature | null;
  onClearSelection: () => void;
  onOpenMatrix?: () => void;
  onProbeZone?: (zone: ZoneFeature) => void;
}

export const ZoneInspector: React.FC<ZoneInspectorProps> = ({
  selectedZone,
  onClearSelection,
  onOpenMatrix,
  onProbeZone,
}) => {
  if (!selectedZone) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center text-slate-500">
        <Crosshair className="w-10 h-10 mb-3 stroke-[1.5] text-slate-600 animate-pulse" />
        <h3 className="text-sm font-semibold text-slate-300">No Zone Selected</h3>
        <p className="text-xs mt-1 max-w-[220px] text-slate-400">
          Click any polygon zone or heatmap anomaly on the map, or select from the zone matrix to inspect pre-visual risk attribution.
        </p>
        {onOpenMatrix && (
          <button
            onClick={onOpenMatrix}
            className="mt-4 px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-400 border border-slate-700 transition-colors"
          >
            Browse Zone Matrix
          </button>
        )}
      </div>
    );
  }

  const p: ZoneProperties = selectedZone.properties;

  const getClassBadge = () => {
    switch (p.risk_class) {
      case 0:
        return {
          label: 'Healthy Canopy',
          color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          icon: <Leaf className="w-3.5 h-3.5" />,
        };
      case 1:
        return {
          label: 'Pre-Visual Stress (Class 1)',
          color: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
          icon: <Sparkles className="w-3.5 h-3.5" />,
        };
      case 2:
        return {
          label: 'High Blight Risk (Class 2)',
          color: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
          icon: <AlertCircle className="w-3.5 h-3.5" />,
        };
      case 3:
        return {
          label: 'Visible Symptoms (Class 3)',
          color: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
          icon: <Flame className="w-3.5 h-3.5" />,
        };
    }
  };

  const getIndicatorExplanation = () => {
    switch (p.dominant_indicator) {
      case 'red_edge_shift':
        return {
          title: 'Red-Edge Blue Shift (705 nm)',
          desc: 'Hyperspectral inflection shifted towards shorter wavelengths, indicating cellular structural degradation before necrosis occurs.',
          confidence: '94.2%',
        };
      case 'pri_decline':
        return {
          title: 'PRI Xanthophyll Cycle Decline (531–570 nm)',
          desc: 'Photochemical Reflectance Index deficit detected. Photosynthetic light-use efficiency drop confirms acute fungal pathogen invasion.',
          confidence: '91.8%',
        };
      case 'chlorophyll_loss':
        return {
          title: 'Canopy Chlorophyll Degradation (680 nm)',
          desc: 'Rapid breakdown of chlorophyll a/b pigments in upper mesophyll layers.',
          confidence: '96.5%',
        };
      case 'water_stress':
        return {
          title: 'SWIR Equivalent Water Thickness Loss (970/1200 nm)',
          desc: 'Water-band index absorption weakening caused by vascular disruption.',
          confidence: '88.4%',
        };
      default:
        return {
          title: 'Canopy Spectral Baseline',
          desc: 'Normal vegetative spectral envelope matching healthy phenology.',
          confidence: '99.0%',
        };
    }
  };

  const badge = getClassBadge();
  const indicator = getIndicatorExplanation();

  return (
    <div className="h-full flex flex-col overflow-y-auto p-4 space-y-4 select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-semibold border border-slate-700">
              {p.zone_id}
            </span>
            <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border ${badge.color}`}>
              {badge.icon}
              {badge.label}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">
            Contract C4 Polygon &bull; {p.area_acres} Acres
          </p>
        </div>
        <button
          onClick={onClearSelection}
          className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Onset Lead-Time Countdown Gauge */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 flex items-center gap-1.5 font-medium">
            <Clock className="w-3.5 h-3.5 text-emerald-400" />
            Forecast Lead-Time Window
          </span>
          <span className="font-mono font-bold text-emerald-400 text-sm">
            {p.days_to_onset.toFixed(1)} Days
          </span>
        </div>
        
        {/* Visual Progress Bar */}
        <div className="mt-2.5 w-full bg-slate-800 h-2 rounded-full overflow-hidden">
          <div 
            className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500"
            style={{ width: `${Math.max(5, (p.days_to_onset / 30) * 100)}%` }}
          />
        </div>
        <p className="mt-1.5 text-[11px] text-slate-400">
          {p.days_to_onset > 0 ? (
            <span className="text-emerald-400 font-medium">
              &bull; Proactive window open: treat before day 0 visible symptom onset
            </span>
          ) : (
            <span className="text-rose-400 font-medium">
              &bull; Visible disease symptoms present on canopy
            </span>
          )}
        </p>
      </div>

      {/* Risk Confidence Score */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 font-medium">Model Risk Probability</span>
          <span className="font-mono font-bold text-white text-sm">
            {(p.risk_score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="mt-2 w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full ${
              p.risk_score > 0.8 ? 'bg-orange-500' : p.risk_score > 0.5 ? 'bg-amber-400' : 'bg-emerald-500'
            }`}
            style={{ width: `${p.risk_score * 100}%` }}
          />
        </div>
      </div>

      {/* Explainability / Integrated Gradients Attribution */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-indigo-400" />
            Attributed Spectral Signature
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            P2 Explainability
          </span>
        </div>
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
          <div className="text-xs font-semibold text-indigo-300">
            {indicator.title}
          </div>
          <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
            {indicator.desc}
          </p>
        </div>
      </div>

      {/* Actionable Agronomic Prescription */}
      <div className="bg-gradient-to-br from-slate-900 to-emerald-950/30 border border-emerald-500/30 rounded-xl p-3.5 space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
          <ShieldAlert className="w-4 h-4 text-emerald-400" />
          <span>Agronomic Prescription</span>
        </div>
        <p className="text-xs text-slate-200 leading-relaxed font-medium bg-slate-950/60 p-2.5 rounded-lg border border-emerald-500/20">
          {p.recommended_action}
        </p>
        <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
          <span>Target Rate: Variable Rate (VRA)</span>
          <span className="text-emerald-400 font-mono">Precision Grid</span>
        </div>
      </div>

      {/* Interactive Hyperspectral Pixel Probe Button */}
      {onProbeZone && (
        <button
          onClick={() => onProbeZone(selectedZone)}
          className="w-full py-2.5 px-3 rounded-xl bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-500/50 text-xs font-semibold flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(6,182,212,0.15)] transition-all hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]"
        >
          <Crosshair className="w-4 h-4 text-cyan-400" />
          <span>Probe Zone Centroid Spectrum (200 Bands)</span>
        </button>
      )}
    </div>
  );
};
