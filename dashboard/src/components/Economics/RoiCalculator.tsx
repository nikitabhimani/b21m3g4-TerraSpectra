import React, { useState } from 'react';
import { DollarSign, Droplets, Leaf, Percent, TrendingUp } from 'lucide-react';

interface RoiCalculatorProps {
  atRiskAcres: number;
  totalAcres: number;
}

export const RoiCalculator: React.FC<RoiCalculatorProps> = ({
  atRiskAcres,
  totalAcres,
}) => {
  const [cropPricePerAcre, setCropPricePerAcre] = useState(1250); // $1250 / acre expected harvest value

  const totalValueAtRisk = Math.round(atRiskAcres * cropPricePerAcre);
  const blanketChemicalCost = Math.round(totalAcres * 14.5); // $14.5/acre blanket
  const targetedChemicalCost = Math.round(atRiskAcres * 22.0); // $22/acre precision VRA
  const chemicalSavings = Math.max(0, blanketChemicalCost - targetedChemicalCost);
  const chemicalSavedPct = Math.round((chemicalSavings / Math.max(1, blanketChemicalCost)) * 100);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 select-none space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-emerald-400" />
          <h4 className="text-xs font-semibold text-white">
            Agronomic Economic ROI & Savings
          </h4>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
          3-Week Pre-Visual Window
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        {/* Yield Loss Avoided */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-emerald-500/20">
          <div className="text-[10px] text-slate-400 font-medium flex items-center justify-between">
            <span>Crop Value Protected</span>
            <TrendingUp className="w-3 h-3 text-emerald-400" />
          </div>
          <div className="text-base font-bold text-white mt-1 font-mono">
            ${totalValueAtRisk.toLocaleString()}
          </div>
          <div className="text-[10px] text-emerald-400 mt-0.5">
            {atRiskAcres.toFixed(1)} acres blight prevented
          </div>
        </div>

        {/* Chemical Input Reduction */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-cyan-500/20">
          <div className="text-[10px] text-slate-400 font-medium flex items-center justify-between">
            <span>Chemical Reduction</span>
            <Droplets className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="text-base font-bold text-cyan-300 mt-1 font-mono">
            {chemicalSavedPct}% Saved
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            ${chemicalSavings.toLocaleString()} input cost saved
          </div>
        </div>
      </div>

      <div className="pt-1 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800">
        <span>Targeted VRA Spray: <b className="text-slate-200">${targetedChemicalCost}</b></span>
        <span>Blanket Spray: <del className="text-slate-500">${blanketChemicalCost}</del></span>
      </div>
    </div>
  );
};
