import React from 'react';
import { AlertTriangle, Clock, ShieldCheck, TrendingUp, Zap } from 'lucide-react';
import { ZoneCollection } from '../types';

interface KpiCardsProps {
  zones: ZoneCollection;
  totalAcres: number;
}

export const KpiCards: React.FC<KpiCardsProps> = ({ zones, totalAcres }) => {
  const features = zones.features || [];
  
  // Calculate summary metrics
  let earlyStressAcres = 0;
  let highRiskAcres = 0;
  let visibleDiseaseAcres = 0;
  let earliestOnset = 30.0;

  features.forEach((f) => {
    const p = f.properties;
    if (p.risk_class === 1) earlyStressAcres += p.area_acres;
    if (p.risk_class === 2) highRiskAcres += p.area_acres;
    if (p.risk_class === 3) visibleDiseaseAcres += p.area_acres;
    if (p.risk_class > 0 && p.days_to_onset < earliestOnset && p.days_to_onset > 0) {
      earliestOnset = p.days_to_onset;
    }
  });

  const totalAtRiskAcres = earlyStressAcres + highRiskAcres + visibleDiseaseAcres;
  const healthyAcres = Math.max(0, totalAcres - totalAtRiskAcres);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 bg-slate-900/60 border-b border-slate-800/80">
      {/* Total Farm Coverage */}
      <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col justify-between hover:border-slate-700 transition-colors">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-xs font-medium">Monitored Area</span>
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
        </div>
        <div className="mt-2 flex items-baseline gap-1.5">
          <span className="text-xl font-bold text-white tracking-tight">
            {totalAcres.toLocaleString()}
          </span>
          <span className="text-xs text-slate-400 font-medium">Acres</span>
        </div>
        <div className="mt-1 text-[11px] text-emerald-400/90 font-medium flex items-center gap-1">
          <span>{healthyAcres.toFixed(1)} ac (98.1%) Vigorous</span>
        </div>
      </div>

      {/* Pre-Visual Stress Detected */}
      <div className="bg-slate-950/80 border border-amber-500/20 rounded-xl p-3 flex flex-col justify-between hover:border-amber-500/40 transition-colors relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-xl group-hover:bg-amber-500/10 transition-all pointer-events-none" />
        <div className="flex items-center justify-between text-amber-300">
          <span className="text-xs font-medium">Pre-Visual Stress</span>
          <Zap className="w-4 h-4 text-amber-400" />
        </div>
        <div className="mt-2 flex items-baseline gap-1.5">
          <span className="text-xl font-bold text-amber-400 tracking-tight">
            {earlyStressAcres.toFixed(1)}
          </span>
          <span className="text-xs text-slate-400 font-medium">Acres</span>
        </div>
        <div className="mt-1 text-[11px] text-amber-400/90 font-medium flex items-center gap-1">
          <TrendingUp className="w-3 h-3" />
          <span>Hidden to human eye (Class 1)</span>
        </div>
      </div>

      {/* High Blight Risk Hotspots */}
      <div className="bg-slate-950/80 border border-orange-500/20 rounded-xl p-3 flex flex-col justify-between hover:border-orange-500/40 transition-colors relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-orange-500/5 rounded-full blur-xl group-hover:bg-orange-500/10 transition-all pointer-events-none" />
        <div className="flex items-center justify-between text-orange-300">
          <span className="text-xs font-medium">High Blight Risk</span>
          <AlertTriangle className="w-4 h-4 text-orange-400" />
        </div>
        <div className="mt-2 flex items-baseline gap-1.5">
          <span className="text-xl font-bold text-orange-400 tracking-tight">
            {highRiskAcres.toFixed(1)}
          </span>
          <span className="text-xs text-slate-400 font-medium">Acres</span>
        </div>
        <div className="mt-1 text-[11px] text-orange-400/90 font-medium flex items-center gap-1">
          <span>Priority spray prescription</span>
        </div>
      </div>

      {/* Forecast Lead Time Window */}
      <div className="bg-slate-950/80 border border-emerald-500/20 rounded-xl p-3 flex flex-col justify-between hover:border-emerald-500/40 transition-colors">
        <div className="flex items-center justify-between text-emerald-300">
          <span className="text-xs font-medium">Forecast Lead Time</span>
          <Clock className="w-4 h-4 text-emerald-400" />
        </div>
        <div className="mt-2 flex items-baseline gap-1.5">
          <span className="text-xl font-bold text-white tracking-tight">
            ~{earliestOnset.toFixed(1)}
          </span>
          <span className="text-xs text-slate-400 font-medium">Days Advance</span>
        </div>
        <div className="mt-1 text-[11px] text-emerald-400 font-medium flex items-center gap-1">
          <span>Proactive treatment window</span>
        </div>
      </div>
    </div>
  );
};
