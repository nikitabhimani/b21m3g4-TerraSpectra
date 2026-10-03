import React, { useState } from 'react';
import { 
  AlertCircle,
  ArrowLeft,
  Calendar,
  Check, 
  Clock,
  Copy, 
  Download, 
  Droplets,
  FileCode, 
  FileSpreadsheet, 
  FileText,
  Flame,
  Info,
  Leaf,
  Plane,
  Printer, 
  ShieldAlert,
  Sparkles,
  UserCheck,
  X 
} from 'lucide-react';
import { ZoneCollection, ZoneFeature } from '../types';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  zones: ZoneCollection;
  fieldName?: string;
  cropType?: string;
  totalAcres?: number;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  zones,
  fieldName = 'Bhatinda North Parcel A',
  cropType = 'Wheat (Triticum aestivum)',
  totalAcres = 1000,
}) => {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'menu' | 'report'>('menu');

  if (!isOpen) return null;

  const features = zones.features || [];

  // Summary calculations
  const atRiskFeatures = features.filter((f) => f.properties.risk_class > 0);
  const atRiskAcres = atRiskFeatures.reduce((acc, f) => acc + f.properties.area_acres, 0);
  const earliestOnset = features.length > 0
    ? Math.min(...features.map((f) => f.properties.days_to_onset))
    : 0;
  const projectedLossPrevented = Math.round(atRiskAcres * 680 * 0.85); // Standard yield salvage model

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
      'prescribed_compound',
      'vra_drone_dosage',
    ];
    const rows = features.map((f) => {
      const p = f.properties;
      const compound = getPrescribedCompound(p.risk_class);
      const droneDosage = getDroneApplicationRate(p.risk_class);
      return [
        p.zone_id,
        p.risk_class,
        p.risk_class_name,
        (p.risk_score * 100).toFixed(1),
        p.area_acres,
        p.days_to_onset.toFixed(1),
        `"${p.dominant_indicator}"`,
        `"${p.recommended_action.replace(/"/g, '""')}"`,
        `"${compound.replace(/"/g, '""')}"`,
        `"${droneDosage.replace(/"/g, '""')}"`,
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
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(JSON.stringify(zones, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handlePrint = () => {
    if (typeof window !== 'undefined' && window.print) {
      window.print();
    }
  };

  function getPrescribedCompound(riskClass: number): string {
    switch (riskClass) {
      case 2:
        return 'Azoxystrobin 250 SC (0.8 L/ha) + Difenoconazole (0.5 L/ha)';
      case 1:
        return 'Bacillus subtilis QST 713 (2.0 L/ha) + Copper Hydroxide (1.5 kg/ha)';
      case 3:
        return 'Curative systemic triazole (Prothioconazole / Tebuconazole @ 1.0 L/ha)';
      default:
        return 'No chemical application. Maintain baseline multispectral monitoring.';
    }
  }

  function getDroneApplicationRate(riskClass: number): string {
    switch (riskClass) {
      case 2:
        return 'UAV VRA: 22.0 L/ha carrier volume, 110-02 nozzles, coarse droplets, 3.0 m altitude';
      case 1:
        return 'UAV VRA: 18.0 L/ha carrier volume, 110-015 nozzles, medium droplets, 3.5 m altitude';
      case 3:
        return 'Ground Rig: 200 L/ha high-volume canopy drench, 3.5 bar';
      default:
        return 'Exclusion Zone (0 L/ha)';
    }
  }

  function getRiskBadge(riskClass: number, name: string) {
    switch (riskClass) {
      case 2:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30">
            <AlertCircle className="w-3 h-3" />
            High Blight Risk
          </span>
        );
      case 1:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
            <Sparkles className="w-3 h-3" />
            Early Stress
          </span>
        );
      case 3:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">
            <Flame className="w-3 h-3" />
            Visible Symptoms
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            <Leaf className="w-3 h-3" />
            Healthy
          </span>
        );
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in select-none overflow-y-auto">
      {viewMode === 'menu' ? (
        /* Standard Export Options Menu */
        <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden my-auto">
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
              aria-label="Close export modal"
              className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-3">
            {/* Printable PDF Spray Plan Option */}
            <div
              onClick={() => setViewMode('report')}
              className="p-3.5 rounded-xl bg-slate-950 border border-emerald-500/40 hover:border-emerald-400 hover:bg-emerald-950/20 transition-all cursor-pointer flex items-center justify-between group shadow-sm shadow-emerald-500/10"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400">
                  <Printer className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <h4 className="text-xs font-bold text-white group-hover:text-emerald-300 transition-colors">
                      Agronomic Spray-Plan & Prescription (PDF / Print)
                    </h4>
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      Recommended
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Field prescription report with UAV drone dosages, chemical specs & CCA sign-off
                  </p>
                </div>
              </div>
              <Printer className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
            </div>

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
                    Contract C4 GeoJSON Polygons (.geojson)
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
              className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-teal-500/50 hover:bg-slate-950/80 transition-all cursor-pointer flex items-center justify-between group"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-white group-hover:text-teal-300 transition-colors">
                    Variable-Rate Spray Plan (.csv)
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
      ) : (
        /* Full Agronomic Printable Spray-Plan Report View */
        <div className="print-document-container w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden my-4">
          {/* Action Toolbar (Hidden during print) */}
          <div className="hide-on-print px-6 py-3.5 border-b border-slate-800 bg-slate-950 flex items-center justify-between">
            <button
              onClick={() => setViewMode('menu')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Export Options</span>
            </button>

            <div className="flex items-center gap-2">
              <button
                onClick={handleDownloadCSV}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download CSV</span>
              </button>
              <button
                onClick={handlePrint}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-600/30 transition-all flex items-center gap-1.5 active:scale-95"
              >
                <Printer className="w-3.5 h-3.5" />
                <span>Print / Save as PDF</span>
              </button>
              <button
                onClick={onClose}
                aria-label="Close report view"
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Printable Document Body */}
          <div className="p-8 space-y-6 max-h-[82vh] overflow-y-auto print:max-h-none print:overflow-visible bg-slate-900 print:bg-white print:text-slate-900">
            {/* Report Header */}
            <div className="border-b border-slate-800 print:border-slate-300 pb-5">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 print:bg-emerald-100 print:text-emerald-800 border border-emerald-500/30">
                      TERRASPECTRA &bull; AEROSPACE ANALYTICS
                    </span>
                    <span className="text-xs text-slate-400 print:text-slate-500">
                      Contract C4 Field Directive
                    </span>
                  </div>
                  <h1 className="text-xl font-bold text-white print:text-slate-900 mt-2">
                    VARIABLE-RATE SPRAY PLAN & AGRONOMIC PRESCRIPTION
                  </h1>
                  <p className="text-xs text-slate-400 print:text-slate-600 mt-0.5">
                    Pre-Visual Hyperspectral Pathogen Forecast & Variable-Rate Application (VRA) Directive
                  </p>
                </div>

                <div className="text-right text-xs font-mono text-slate-400 print:text-slate-600">
                  <div>Date: {new Date(zones.generated_at).toLocaleDateString()}</div>
                  <div>Time: {new Date(zones.generated_at).toLocaleTimeString()}</div>
                  <div className="text-emerald-400 print:text-emerald-700 font-bold mt-1">STATUS: CERTIFIED</div>
                </div>
              </div>

              {/* Metadata Grid */}
              <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 p-3.5 rounded-xl bg-slate-950/70 print:bg-slate-50 border border-slate-800 print:border-slate-200 text-xs">
                <div>
                  <span className="text-slate-500 print:text-slate-400 block text-[10px] uppercase font-mono">Parcel Name</span>
                  <span className="font-semibold text-slate-200 print:text-slate-900">{fieldName}</span>
                </div>
                <div>
                  <span className="text-slate-500 print:text-slate-400 block text-[10px] uppercase font-mono">Crop Variety</span>
                  <span className="font-semibold text-slate-200 print:text-slate-900">{cropType}</span>
                </div>
                <div>
                  <span className="text-slate-500 print:text-slate-400 block text-[10px] uppercase font-mono">Monitored Area</span>
                  <span className="font-semibold text-slate-200 print:text-slate-900">{totalAcres} Acres</span>
                </div>
                <div>
                  <span className="text-slate-500 print:text-slate-400 block text-[10px] uppercase font-mono">Job Reference</span>
                  <span className="font-mono text-emerald-400 print:text-emerald-700 text-[11px] truncate block">
                    {zones.job_id}
                  </span>
                </div>
              </div>
            </div>

            {/* Executive Scorecard */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-500 block">Total At-Risk Acreage</span>
                <span className="text-lg font-bold text-amber-400 print:text-amber-700 font-mono mt-0.5 block">
                  {atRiskAcres.toFixed(1)} ac
                </span>
                <span className="text-[10px] text-slate-500">
                  {((atRiskAcres / totalAcres) * 100).toFixed(1)}% of monitored field
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-500 block">Earliest Visible Onset</span>
                <span className="text-lg font-bold text-emerald-400 print:text-emerald-700 font-mono mt-0.5 block">
                  ~{earliestOnset.toFixed(1)} Days
                </span>
                <span className="text-[10px] text-slate-500">
                  Pre-visual intervention window
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-500 block">Protected Crop Value</span>
                <span className="text-lg font-bold text-teal-400 print:text-teal-700 font-mono mt-0.5 block">
                  ${projectedLossPrevented.toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-500">
                  Projected disease loss prevented
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-500 block">Model Architecture</span>
                <span className="text-xs font-bold text-indigo-400 print:text-indigo-700 mt-1 block">
                  3D-CNN + ViT Hybrid
                </span>
                <span className="text-[10px] text-slate-500">
                  200 channels (400–2500 nm)
                </span>
              </div>
            </div>

            {/* Diagnostic Spectral Signatures */}
            <div className="p-4 rounded-xl bg-slate-950/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 space-y-2.5 print-page-break-inside-avoid">
              <h3 className="text-xs font-bold text-slate-200 print:text-slate-800 flex items-center gap-1.5 uppercase tracking-wide">
                <Info className="w-3.5 h-3.5 text-emerald-400 print:text-emerald-600" />
                <span>Pre-Visual Diagnostic Hyperspectral Indicators</span>
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-400 print:text-slate-600">
                <div className="p-2.5 rounded-lg bg-slate-900 print:bg-white border border-slate-800/80 print:border-slate-200">
                  <span className="font-semibold text-slate-200 print:text-slate-800 block text-[11px]">
                    &bull; Red-Edge Position (REP 705–750 nm)
                  </span>
                  <p className="text-[11px] mt-0.5">
                    Inflection wavelength blueshift indicates early mesophyll cellular collapse 14–21 days prior to macroscopic foliar yellowing.
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900 print:bg-white border border-slate-800/80 print:border-slate-200">
                  <span className="font-semibold text-slate-200 print:text-slate-800 block text-[11px]">
                    &bull; Photochemical Reflectance Index (PRI 531/570 nm)
                  </span>
                  <p className="text-[11px] mt-0.5">
                    Xanthophyll cycle de-epoxidation deficit reveals acute photosynthetic inhibition caused by initial fungal mycelium penetration.
                  </p>
                </div>
              </div>
            </div>

            {/* Tabular Prescription Plan */}
            <div className="space-y-3 print-page-break-inside-avoid">
              <h3 className="text-xs font-bold text-slate-200 print:text-slate-800 uppercase tracking-wide flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-emerald-400 print:text-emerald-600" />
                <span>Zone-by-Zone Variable-Rate Application (VRA) Prescription</span>
              </h3>

              <div className="border border-slate-800 print:border-slate-300 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-950 print:bg-slate-100 border-b border-slate-800 print:border-slate-300 text-slate-400 print:text-slate-700 font-mono text-[11px]">
                      <th className="p-3">Zone ID</th>
                      <th className="p-3">Risk Category</th>
                      <th className="p-3">Area</th>
                      <th className="p-3">Onset Lead</th>
                      <th className="p-3">Prescribed Compound & Dosage</th>
                      <th className="p-3">UAV Drone Application Specs</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 print:divide-slate-200">
                    {features.map((f) => {
                      const p = f.properties;
                      return (
                        <tr key={p.zone_id} className="hover:bg-slate-800/30 print:hover:bg-transparent">
                          <td className="p-3 font-mono font-bold text-white print:text-slate-900">
                            {p.zone_id}
                          </td>
                          <td className="p-3">
                            {getRiskBadge(p.risk_class, p.risk_class_name)}
                          </td>
                          <td className="p-3 font-mono text-slate-300 print:text-slate-800">
                            {p.area_acres.toFixed(1)} ac
                          </td>
                          <td className="p-3 font-mono text-emerald-400 print:text-emerald-700 font-semibold">
                            {p.days_to_onset.toFixed(1)} d
                          </td>
                          <td className="p-3 text-slate-200 print:text-slate-800 text-[11px]">
                            <div className="font-semibold">{getPrescribedCompound(p.risk_class)}</div>
                            <div className="text-[10px] text-slate-400 print:text-slate-500 mt-0.5">
                              Action: {p.recommended_action}
                            </div>
                          </td>
                          <td className="p-3 text-slate-400 print:text-slate-600 text-[10px] font-mono">
                            {getDroneApplicationRate(p.risk_class)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* UAV Flight & Environmental Constraints */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs print-page-break-inside-avoid">
              <div className="p-3.5 rounded-xl bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200 space-y-1.5">
                <span className="font-semibold text-slate-200 print:text-slate-800 flex items-center gap-1.5">
                  <Plane className="w-3.5 h-3.5 text-teal-400 print:text-teal-600" />
                  <span>UAV Flight Parameters</span>
                </span>
                <ul className="text-[11px] text-slate-400 print:text-slate-600 space-y-1 list-disc list-inside">
                  <li>Flight altitude: 3.0 m – 3.5 m above crop canopy</li>
                  <li>Ground speed: 18.0 km/h (optimal spray droplet deposition)</li>
                  <li>Nozzle configuration: Anti-drift air-induction coarse tips</li>
                  <li>Maintain 15-meter spray exclusion buffer near field drainage</li>
                </ul>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200 space-y-1.5">
                <span className="font-semibold text-slate-200 print:text-slate-800 flex items-center gap-1.5">
                  <Droplets className="w-3.5 h-3.5 text-emerald-400 print:text-emerald-600" />
                  <span>Environmental Application Limits</span>
                </span>
                <ul className="text-[11px] text-slate-400 print:text-slate-600 space-y-1 list-disc list-inside">
                  <li>Max wind speed: 12 km/h (avoid droplet drift)</li>
                  <li>Target temperature: 15°C – 24°C</li>
                  <li>Relative humidity: &gt; 60% (promotes bio-agent foliar colonization)</li>
                  <li>Re-entry interval (REI): 12 hours post-application</li>
                </ul>
              </div>
            </div>

            {/* Certified Agronomist Sign-Off Block */}
            <div className="p-4 rounded-xl bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-300 print-page-break-inside-avoid">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200 print:text-slate-800 uppercase tracking-wide mb-3">
                <UserCheck className="w-4 h-4 text-emerald-400 print:text-emerald-600" />
                <span>Certified Crop Advisor (CCA) Verification & Field Sign-Off</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-2 text-xs text-slate-400 print:text-slate-600">
                <div>
                  <div className="border-b border-slate-700 print:border-slate-400 pb-1 h-6"></div>
                  <span className="text-[10px] uppercase font-mono block mt-1">Lead Agronomist Signature</span>
                </div>
                <div>
                  <div className="border-b border-slate-700 print:border-slate-400 pb-1 h-6"></div>
                  <span className="text-[10px] uppercase font-mono block mt-1">CCA License Number</span>
                </div>
                <div>
                  <div className="border-b border-slate-700 print:border-slate-400 pb-1 h-6"></div>
                  <span className="text-[10px] uppercase font-mono block mt-1">UAV Pilot Remote Certificate ID</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
