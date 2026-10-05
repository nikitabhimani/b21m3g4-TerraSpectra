import React, { useState } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Check, 
  Copy, 
  Crosshair, 
  Download, 
  Droplet, 
  FileText, 
  HelpCircle, 
  Info, 
  Layers, 
  Maximize2, 
  Radio, 
  ShieldAlert, 
  Sparkles, 
  Target, 
  X 
} from 'lucide-react';
import { PixelProbeData, PixelSpectrumBand } from '../types';
import { exportProbeToJson, exportSpectrumToCsv } from '../services/pixelProbe';

interface PixelProbeModalProps {
  isOpen: boolean;
  onClose: () => void;
  probeData: PixelProbeData | null;
}

export const PixelProbeModal: React.FC<PixelProbeModalProps> = ({
  isOpen,
  onClose,
  probeData,
}) => {
  const [hoveredBand, setHoveredBand] = useState<PixelSpectrumBand | null>(null);
  const [copiedGps, setCopiedGps] = useState<boolean>(false);
  const [copiedJson, setCopiedJson] = useState<boolean>(false);
  const [activeCurveToggle, setActiveCurveToggle] = useState<{
    healthy: boolean;
    stressed: boolean;
    soil: boolean;
  }>({
    healthy: true,
    stressed: true,
    soil: false,
  });

  if (!isOpen || !probeData) return null;

  const { indices, spectrum } = probeData;

  const handleCopyGps = () => {
    navigator.clipboard.writeText(`${probeData.lat}, ${probeData.lng}`);
    setCopiedGps(true);
    setTimeout(() => setCopiedGps(false), 2000);
  };

  const handleCopyJson = () => {
    const jsonStr = exportProbeToJson(probeData);
    navigator.clipboard.writeText(jsonStr);
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handleDownloadCsv = () => {
    const csvContent = exportSpectrumToCsv(probeData);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `terraspectra_pixel_${probeData.lat}_${probeData.lng}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // SVG Chart Geometry
  const width = 760;
  const height = 230;
  const padding = { top: 25, right: 30, bottom: 40, left: 50 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const minWl = 400;
  const maxWl = 2500;
  const minVal = 0.0;
  const maxVal = 0.65;

  const getX = (wl: number) => padding.left + ((wl - minWl) / (maxWl - minWl)) * chartW;
  const getY = (val: number) => padding.top + chartH - ((val - minVal) / (maxVal - minVal)) * chartH;

  // Path generators
  const buildPath = (accessor: (b: PixelSpectrumBand) => number) => {
    return spectrum
      .map((b, i) => `${i === 0 ? 'M' : 'L'} ${getX(b.wavelength).toFixed(1)} ${getY(accessor(b)).toFixed(1)}`)
      .join(' ');
  };

  const sampledPath = buildPath((b) => b.reflectance);
  const healthyPath = buildPath((b) => b.healthyReflectance);
  const stressedPath = buildPath((b) => b.stressedReflectance);
  const soilPath = buildPath((b) => b.soilReflectance);

  // Status Styling
  const getBadgeStyle = () => {
    switch (probeData.risk_class) {
      case 0:
        return {
          bg: 'bg-emerald-950/70 border-emerald-500/40 text-emerald-300',
          dot: 'bg-emerald-400',
          title: 'Optimal Photosynthetic Canopy',
        };
      case 1:
        return {
          bg: 'bg-amber-950/70 border-amber-500/40 text-amber-300',
          dot: 'bg-amber-400',
          title: 'Early Pre-Visual Stress Anomaly',
        };
      case 2:
        return {
          bg: 'bg-orange-950/70 border-orange-500/40 text-orange-300',
          dot: 'bg-orange-400',
          title: 'High Blight Pathogenesis Risk',
        };
      case 3:
        return {
          bg: 'bg-rose-950/70 border-rose-500/40 text-rose-300',
          dot: 'bg-rose-400',
          title: 'Visible Chlorosis / Necrotic Lesion',
        };
      default:
        return {
          bg: 'bg-slate-900 border-slate-700 text-slate-300',
          dot: 'bg-slate-400',
          title: 'Unclassified Soil/Boundary',
        };
    }
  };

  const badge = getBadgeStyle();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div 
        className="w-full max-w-4xl max-h-[92vh] bg-slate-900/95 border border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden text-slate-100 animate-in fade-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pixel-probe-title"
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.25)]">
              <Crosshair className="w-5 h-5 animate-spin-slow" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 id="pixel-probe-title" className="text-sm font-bold text-white tracking-wide">
                  Hyperspectral Pixel Probe Telemetry
                </h3>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  {probeData.sensorId}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono mt-0.5">
                <span>{probeData.field_name}</span>
                <span>&bull;</span>
                <span className="text-slate-300">
                  {probeData.lat}° N, {probeData.lng}° E
                </span>
                <span>&bull;</span>
                <span>Elev: {probeData.elevationMeters}m</span>
                <span>&bull;</span>
                <span>GSD: {probeData.gsdMeters}m</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyGps}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5 transition-colors"
              title="Copy GPS Coordinates"
            >
              {copiedGps ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
              <span>{copiedGps ? 'Copied!' : 'Copy GPS'}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              aria-label="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Top Diagnostic Banner */}
          <div className={`p-4 rounded-xl border flex flex-col md:flex-row md:items-center justify-between gap-4 ${badge.bg}`}>
            <div className="flex items-center gap-3">
              <span className={`w-3 h-3 rounded-full animate-ping ${badge.dot}`} />
              <div>
                <div className="text-xs uppercase font-mono tracking-wider text-slate-400">
                  Pixel Classification & Risk Status
                </div>
                <div className="text-base font-bold text-white flex items-center gap-2">
                  <span>{badge.title}</span>
                  {probeData.nearestZoneId && (
                    <span className="text-xs font-normal text-slate-300">
                      ({probeData.zone_name})
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 text-xs font-mono">
              <div>
                <div className="text-slate-400 text-[10px]">RISK SCORE</div>
                <div className="text-sm font-bold text-white">
                  {(probeData.risk_score * 100).toFixed(1)}%
                </div>
              </div>
              <div className="border-l border-slate-700/60 pl-4">
                <div className="text-slate-400 text-[10px]">EST. ONSET LEAD TIME</div>
                <div className="text-sm font-bold text-emerald-400">
                  ~{probeData.days_to_onset.toFixed(1)} Days Ahead
                </div>
              </div>
              <div className="border-l border-slate-700/60 pl-4">
                <div className="text-slate-400 text-[10px]">DOMINANT BIOMARKER</div>
                <div className="text-sm font-bold text-cyan-300">
                  {probeData.dominant_indicator.replace('_', ' ').toUpperCase()}
                </div>
              </div>
            </div>
          </div>

          {/* Interactive 200-Band Hyperspectral Reflectance Graph */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 shadow-inner">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h4 className="text-xs font-semibold text-white">
                  Continuous 200-Band Hyperspectral Profile (400–2500 nm)
                </h4>
              </div>

              {/* Curve Toggle Controls */}
              <div className="flex items-center gap-3 text-[11px]">
                <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                  <span className="w-3 h-1 bg-cyan-400 rounded-full shadow-[0_0_8px_#06b6d4]" />
                  Probed Pixel
                </span>
                <button
                  onClick={() => setActiveCurveToggle((prev) => ({ ...prev, healthy: !prev.healthy }))}
                  className={`flex items-center gap-1.5 transition-opacity ${
                    activeCurveToggle.healthy ? 'opacity-100 text-emerald-400' : 'opacity-40 text-slate-500'
                  }`}
                >
                  <span className="w-3 h-0.5 border-t border-dashed border-emerald-400" />
                  Healthy Ref
                </button>
                <button
                  onClick={() => setActiveCurveToggle((prev) => ({ ...prev, stressed: !prev.stressed }))}
                  className={`flex items-center gap-1.5 transition-opacity ${
                    activeCurveToggle.stressed ? 'opacity-100 text-amber-400' : 'opacity-40 text-slate-500'
                  }`}
                >
                  <span className="w-3 h-0.5 border-t border-dashed border-amber-400" />
                  Stressed Ref
                </button>
                <button
                  onClick={() => setActiveCurveToggle((prev) => ({ ...prev, soil: !prev.soil }))}
                  className={`flex items-center gap-1.5 transition-opacity ${
                    activeCurveToggle.soil ? 'opacity-100 text-slate-300' : 'opacity-40 text-slate-600'
                  }`}
                >
                  <span className="w-3 h-0.5 border-t border-dotted border-slate-400" />
                  Soil Baseline
                </button>
              </div>
            </div>

            {/* SVG Interactive Canvas */}
            <div className="relative w-full overflow-hidden flex flex-col items-center">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-auto max-h-[260px] cursor-crosshair"
                onMouseMove={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const mouseX = e.clientX - rect.left;
                  const ratio = (mouseX / rect.width) * width;
                  const targetWl = minWl + ((ratio - padding.left) / chartW) * (maxWl - minWl);
                  if (targetWl >= minWl && targetWl <= maxWl) {
                    const closest = spectrum.reduce((prev, curr) => 
                      Math.abs(curr.wavelength - targetWl) < Math.abs(prev.wavelength - targetWl) ? curr : prev
                    );
                    setHoveredBand(closest);
                  }
                }}
                onMouseLeave={() => setHoveredBand(null)}
              >
                {/* Shaded Diagnostic Spectral Windows */}
                {/* 1. PRI (531nm) */}
                <rect
                  x={getX(520)}
                  y={padding.top}
                  width={getX(570) - getX(520)}
                  height={chartH}
                  fill="#f59e0b"
                  opacity="0.14"
                />
                <text x={getX(545)} y={padding.top + 12} fill="#fbbf24" fontSize="9" textAnchor="middle" fontFamily="monospace">
                  PRI 531nm
                </text>

                {/* 2. Chlorophyll Pit (670nm) */}
                <rect
                  x={getX(650)}
                  y={padding.top}
                  width={getX(685) - getX(650)}
                  height={chartH}
                  fill="#ef4444"
                  opacity="0.12"
                />

                {/* 3. Red-Edge Transition (690-745nm) */}
                <rect
                  x={getX(690)}
                  y={padding.top}
                  width={getX(745) - getX(690)}
                  height={chartH}
                  fill="#6366f1"
                  opacity="0.18"
                />
                <text x={getX(717)} y={padding.top + 12} fill="#818cf8" fontSize="9" textAnchor="middle" fontFamily="monospace">
                  Red-Edge (REP)
                </text>

                {/* 4. Canopy Water Absorption (970nm & 1240nm) */}
                <rect
                  x={getX(940)}
                  y={padding.top}
                  width={getX(1010) - getX(940)}
                  height={chartH}
                  fill="#06b6d4"
                  opacity="0.12"
                />
                <rect
                  x={getX(1200)}
                  y={padding.top}
                  width={getX(1270) - getX(1200)}
                  height={chartH}
                  fill="#06b6d4"
                  opacity="0.12"
                />

                {/* Horizontal Grid lines */}
                {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((val) => (
                  <g key={val}>
                    <line
                      x1={padding.left}
                      y1={getY(val)}
                      x2={width - padding.right}
                      y2={getY(val)}
                      stroke="#1e293b"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padding.left - 8}
                      y={getY(val) + 3}
                      fill="#64748b"
                      fontSize="9"
                      textAnchor="end"
                      fontFamily="monospace"
                    >
                      {(val * 100).toFixed(0)}%
                    </text>
                  </g>
                ))}

                {/* Vertical Wavelength Grid Lines & Labels */}
                {[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500].map((wl) => (
                  <g key={wl}>
                    <line
                      x1={getX(wl)}
                      y1={padding.top}
                      x2={getX(wl)}
                      y2={height - padding.bottom}
                      stroke="#1e293b"
                      strokeDasharray="2 3"
                    />
                    <text
                      x={getX(wl)}
                      y={height - padding.bottom + 16}
                      fill="#64748b"
                      fontSize="9"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      {wl}
                    </text>
                  </g>
                ))}

                {/* Reference Curves */}
                {activeCurveToggle.soil && (
                  <path
                    d={soilPath}
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="1.5"
                    strokeDasharray="2 3"
                    opacity="0.7"
                  />
                )}
                {activeCurveToggle.healthy && (
                  <path
                    d={healthyPath}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="1.5"
                    strokeDasharray="4 3"
                    opacity="0.75"
                  />
                )}
                {activeCurveToggle.stressed && (
                  <path
                    d={stressedPath}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="1.5"
                    strokeDasharray="4 3"
                    opacity="0.75"
                  />
                )}

                {/* Primary Sampled Pixel Curve */}
                <path
                  d={sampledPath}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2.5"
                  className="filter drop-shadow-[0_0_6px_rgba(6,182,212,0.6)]"
                />

                {/* Hover Probe Cursor Line */}
                {hoveredBand && (
                  <g>
                    <line
                      x1={getX(hoveredBand.wavelength)}
                      y1={padding.top}
                      x2={getX(hoveredBand.wavelength)}
                      y2={height - padding.bottom}
                      stroke="#38bdf8"
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                    />
                    <circle
                      cx={getX(hoveredBand.wavelength)}
                      cy={getY(hoveredBand.reflectance)}
                      r="4.5"
                      fill="#06b6d4"
                      stroke="#ffffff"
                      strokeWidth="2"
                    />
                  </g>
                )}
              </svg>

              {/* Hover Telemetry Readout Box */}
              <div className="w-full mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
                <div className="text-slate-400">
                  {hoveredBand ? (
                    <span className="text-cyan-300">
                      Band #{hoveredBand.bandIndex} &bull; Wavelength:{' '}
                      <b className="text-white">{hoveredBand.wavelength} nm</b> &bull; Reflectance:{' '}
                      <b className="text-white">{(hoveredBand.reflectance * 100).toFixed(2)}%</b> &bull; SNR:{' '}
                      <b className="text-emerald-400">{hoveredBand.snr}:1</b>
                    </span>
                  ) : (
                    <span>Hover over chart to inspect single-band reflectance & sensor signal-to-noise ratio</span>
                  )}
                </div>
                <div className="text-slate-500 text-[10px]">Wavelength Grid: Contract C1 Canonical</div>
              </div>
            </div>
          </div>

          {/* Real-Time Spectral Indices Matrix */}
          <div>
            <div className="flex items-center gap-2 mb-2.5">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <h4 className="text-xs font-semibold text-white">Calculated Biophysical Stress Indices</h4>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {/* 1. NDVI */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400">NDVI (840/670)</span>
                  <span className={`w-2 h-2 rounded-full ${indices.ndvi > 0.6 ? 'bg-emerald-400' : indices.ndvi > 0.35 ? 'bg-amber-400' : 'bg-rose-400'}`} />
                </div>
                <div className="text-lg font-bold text-white my-1 font-mono">{indices.ndvi.toFixed(3)}</div>
                <div className="text-[10px] text-slate-400 leading-tight">Canopy Greenness & Biomass</div>
              </div>

              {/* 2. NDRE */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400">NDRE (790/720)</span>
                  <span className={`w-2 h-2 rounded-full ${indices.ndre > 0.4 ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                </div>
                <div className="text-lg font-bold text-white my-1 font-mono">{indices.ndre.toFixed(3)}</div>
                <div className="text-[10px] text-slate-400 leading-tight">Dense Foliage Chlorophyll</div>
              </div>

              {/* 3. PRI (Key Pre-Visual Metric) */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-cyan-400 font-semibold">PRI (531/570)</span>
                  <span className={`w-2 h-2 rounded-full ${indices.pri >= 0.0 ? 'bg-emerald-400' : 'bg-orange-400 shadow-[0_0_6px_#f97316]'}`} />
                </div>
                <div className="text-lg font-bold text-cyan-300 my-1 font-mono">{indices.pri.toFixed(3)}</div>
                <div className="text-[10px] text-slate-400 leading-tight">Xanthophyll Cycle Efficiency</div>
              </div>

              {/* 4. Red-Edge Position (Guyot-Baret) */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-indigo-400 font-semibold">REP (Guyot)</span>
                  <span className={`w-2 h-2 rounded-full ${indices.rep > 718 ? 'bg-emerald-400' : 'bg-rose-400 shadow-[0_0_6px_#f43f5e]'}`} />
                </div>
                <div className="text-lg font-bold text-indigo-300 my-1 font-mono">{indices.rep.toFixed(1)} <span className="text-xs font-normal">nm</span></div>
                <div className="text-[10px] text-slate-400 leading-tight">
                  {indices.rep < 715 ? 'Diagnostic Blue-Shift!' : 'Optimal Red-Edge'}
                </div>
              </div>

              {/* 5. NDWI */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400">NDWI (860/1240)</span>
                  <span className={`w-2 h-2 rounded-full ${indices.ndwi > 0.1 ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                </div>
                <div className="text-lg font-bold text-white my-1 font-mono">{indices.ndwi.toFixed(3)}</div>
                <div className="text-[10px] text-slate-400 leading-tight">Canopy Cellular Moisture</div>
              </div>

              {/* 6. MCARI */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400">MCARI (700/670)</span>
                  <span className="w-2 h-2 rounded-full bg-slate-400" />
                </div>
                <div className="text-lg font-bold text-white my-1 font-mono">{indices.mcari.toFixed(3)}</div>
                <div className="text-[10px] text-slate-400 leading-tight">Chlorophyll Absorption Depth</div>
              </div>
            </div>
          </div>

          {/* Biophysical Pathology & Recommended Prescription */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
              <h5 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                <span>Biophysical Anomaly Diagnosis</span>
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">
                {probeData.diagnosticNotes}
              </p>
              <ul className="text-[11px] text-slate-400 space-y-1 list-disc list-inside pt-1">
                {probeData.biophysicalMechanisms.map((mech, i) => (
                  <li key={i}>{mech}</li>
                ))}
              </ul>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
              <h5 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Droplet className="w-4 h-4 text-emerald-400" />
                <span>Targeted Agronomic Intervention</span>
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">
                {probeData.recommendedIntervention}
              </p>
              <div className="mt-3 p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/20 text-[11px] text-emerald-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  Actionable 21 days earlier than standard RGB scouting, preventing up to <b>88%</b> of potential harvest yield loss.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer Actions */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/90 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadCsv}
              className="px-3 py-1.5 rounded-xl text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/30 flex items-center gap-1.5 transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download 200-Band Spectrum (.csv)</span>
            </button>
            <button
              onClick={handleCopyJson}
              className="px-3 py-1.5 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5 transition-colors"
            >
              {copiedJson ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <FileText className="w-3.5 h-3.5 text-slate-400" />}
              <span>{copiedJson ? 'JSON Copied!' : 'Copy Telemetry (JSON)'}</span>
            </button>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};
