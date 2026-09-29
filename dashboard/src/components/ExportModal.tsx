import React, { useState } from 'react';
import { 
  Check, 
  Copy, 
  Download, 
  FileCode, 
  FileSpreadsheet, 
  MapPin, 
  Printer, 
  X 
} from 'lucide-react';
import { ZoneCollection } from '../types';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  zones: ZoneCollection;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  zones,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const features = zones.features || [];

  const handleDownloadGeoJSON = () => {
    const blob = new Blob([JSON.stringify(zones, null, 2)], {
      type: 'application/geo+json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terraspectra_c4_zones_${zones.job_id}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadCSV = () => {
    const headers = [
      'zone_id',
      'risk_class',
      'risk_class_name',
      'risk_score_pct',
      'area_acres',
      'days_to_onset',
      'dominant_indicator',
      'recommended_action',
    ];
    const rows = features.map((f) => {
      const p = f.properties;
      return [
        p.zone_id,
        p.risk_class,
        p.risk_class_name,
        (p.risk_score * 100).toFixed(1),
        p.area_acres,
        p.days_to_onset.toFixed(1),
        `"${p.dominant_indicator}"`,
        `"${p.recommended_action.replace(/"/g, '""')}"`,
      ].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terraspectra_spray_plan_${zones.job_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyJSON = () => {
    navigator.clipboard.writeText(JSON.stringify(zones, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in select-none">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Download className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">
                Export Agronomic Forecast
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Job: {zones.job_id}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-3">
          {/* GeoJSON Option */}
          <div
            onClick={handleDownloadGeoJSON}
            className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-emerald-500/50 hover:bg-slate-950/80 transition-all cursor-pointer flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                <FileCode className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-white group-hover:text-emerald-300 transition-colors">
                  Contract C4 GeoJSON Polygons
                </h4>
                <p className="text-[11px] text-slate-400">
                  Import directly into QGIS, ArcGIS, or John Deere Operations Center
                </p>
              </div>
            </div>
            <Download className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 transition-colors" />
          </div>

          {/* CSV Spray Plan Option */}
          <div
            onClick={handleDownloadCSV}
            className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-emerald-500/50 hover:bg-slate-950/80 transition-all cursor-pointer flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-white group-hover:text-teal-300 transition-colors">
                  Variable-Rate Spray Plan (CSV)
                </h4>
                <p className="text-[11px] text-slate-400">
                  Prescription table with zone acreage, onset countdown & chemical action
                </p>
              </div>
            </div>
            <Download className="w-4 h-4 text-slate-500 group-hover:text-teal-400 transition-colors" />
          </div>

          {/* Copy to Clipboard */}
          <div
            onClick={handleCopyJSON}
            className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-slate-700 hover:bg-slate-950/80 transition-all cursor-pointer flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-slate-800 text-slate-300">
                <Copy className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-white group-hover:text-slate-200 transition-colors">
                  Copy GeoJSON Payload
                </h4>
                <p className="text-[11px] text-slate-400">
                  Copy raw FeatureCollection JSON to system clipboard
                </p>
              </div>
            </div>
            {copied ? (
              <Check className="w-4 h-4 text-emerald-400" />
            ) : (
              <Copy className="w-4 h-4 text-slate-500 group-hover:text-slate-300" />
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
