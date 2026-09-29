import React, { useState } from 'react';
import { 
  Compass, 
  Eye, 
  Layers, 
  MapPin, 
  Maximize2, 
  Minimize2, 
  Sliders, 
  SunMedium, 
  ZoomIn, 
  ZoomOut 
} from 'lucide-react';
import { FieldItem, ZoneCollection, ZoneFeature } from '../../types';

interface MapViewProps {
  zones: ZoneCollection;
  activeField: FieldItem;
  selectedZone: ZoneFeature | null;
  onSelectZone: (zone: ZoneFeature) => void;
  daysHorizon: number;
}

export const MapView: React.FC<MapViewProps> = ({
  zones,
  activeField,
  selectedZone,
  onSelectZone,
  daysHorizon,
}) => {
  const [zoom, setZoom] = useState(1);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [basemap, setBasemap] = useState<'satellite' | 'dark' | 'terrain'>('satellite');
  const [hoveredZone, setHoveredZone] = useState<ZoneFeature | null>(null);

  const features = zones.features || [];

  // Filter features according to daysHorizon
  const visibleFeatures = features.filter(
    (f) => f.properties.risk_class === 0 || f.properties.days_to_onset >= (30 - daysHorizon)
  );

  const getColor = (riskClass: number) => {
    switch (riskClass) {
      case 0:
        return { fill: '#10b981', stroke: '#059669', text: 'text-emerald-400' };
      case 1:
        return { fill: '#f59e0b', stroke: '#d97706', text: 'text-amber-400' };
      case 2:
        return { fill: '#f97316', stroke: '#ea580c', text: 'text-orange-400' };
      case 3:
        return { fill: '#ef4444', stroke: '#dc2626', text: 'text-rose-400' };
      default:
        return { fill: '#64748b', stroke: '#475569', text: 'text-slate-400' };
    }
  };

  // Geo coordinate bounds projection to SVG viewBox (0..800 x 0..600)
  const coordsAll = activeField.boundary[0];
  const lats = coordsAll.map((c) => c[1]);
  const lngs = coordsAll.map((c) => c[0]);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);

  const svgW = 800;
  const svgH = 540;
  const pad = 40;

  const project = (lng: number, lat: number): [number, number] => {
    const x = pad + ((lng - minLng) / (maxLng - minLng)) * (svgW - pad * 2);
    // Invert Y for screen coords
    const y = pad + ((maxLat - lat) / (maxLat - minLat)) * (svgH - pad * 2);
    return [x, y];
  };

  const fieldPolygonPath = activeField.boundary[0]
    .map(([lng, lat], i) => {
      const [px, py] = project(lng, lat);
      return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
    })
    .join(' ') + ' Z';

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden flex flex-col select-none">
      {/* Map Header Toolbar */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-slate-800 px-3 py-1.5 rounded-xl shadow-lg">
        <MapPin className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-xs font-semibold text-white tracking-wide">
          {activeField.name}
        </span>
        <span className="text-[11px] text-slate-400 font-mono">
          ({activeField.crop_type})
        </span>
      </div>

      {/* Layer Controls (Top Right) */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
        {/* Basemap Switcher */}
        <div className="flex bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-xl p-1 shadow-lg text-xs">
          <button
            onClick={() => setBasemap('satellite')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              basemap === 'satellite'
                ? 'bg-emerald-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Satellite
          </button>
          <button
            onClick={() => setBasemap('dark')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              basemap === 'dark'
                ? 'bg-emerald-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Dark Grid
          </button>
          <button
            onClick={() => setBasemap('terrain')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              basemap === 'terrain'
                ? 'bg-emerald-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Terrain 3D
          </button>
        </div>

        {/* Heatmap Toggle & Slider */}
        <div className="flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-slate-800 px-3 py-1.5 rounded-xl shadow-lg text-xs">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-1.5 font-medium transition-colors ${
              showHeatmap ? 'text-emerald-400' : 'text-slate-500'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Heatmap</span>
          </button>
          {showHeatmap && (
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              title="Heatmap Opacity"
            />
          )}
        </div>
      </div>

      {/* Interactive Map Canvas Area */}
      <div className="relative flex-1 w-full h-full flex items-center justify-center overflow-hidden">
        {/* Synthetic Satellite / Grid Texture Background */}
        <div 
          className={`absolute inset-0 transition-opacity duration-700 ${
            basemap === 'satellite' 
              ? 'bg-[radial-gradient(#152e22_1px,transparent_1px)] [background-size:16px_16px] bg-[#091510]'
              : basemap === 'dark'
              ? 'bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:20px_20px] bg-slate-950'
              : 'bg-gradient-to-tr from-stone-950 via-slate-900 to-emerald-950'
          }`}
        />

        {/* SVG Vector Map Rendering */}
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full h-full max-h-[85vh] cursor-grab active:cursor-grabbing transition-transform duration-300"
          style={{ transform: `scale(${zoom})` }}
        >
          <defs>
            {/* Heatmap blur filter */}
            <filter id="heatmap-blur" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="16" result="blur" />
            </filter>
            {/* 3D Extrusion drop shadow */}
            <filter id="zone-shadow" x="-10%" y="-10%" width="120%" height="130%">
              <feDropShadow dx="3" dy="6" stdDeviation="4" floodColor="#000000" floodOpacity="0.6" />
            </filter>
          </defs>

          {/* Farm Boundary (Contract C3 Field Boundary) */}
          <path
            d={fieldPolygonPath}
            fill="#064e3b"
            fillOpacity={basemap === 'satellite' ? '0.22' : '0.12'}
            stroke="#10b981"
            strokeWidth="1.5"
            strokeDasharray="6 4"
            className="transition-all"
          />

          {/* Heatmap Layer Simulation (Contract C3 /tiles) */}
          {showHeatmap && (
            <g filter="url(#heatmap-blur)" opacity={heatmapOpacity}>
              {visibleFeatures
                .filter((f) => f.properties.risk_class > 0)
                .map((f) => {
                  const ring = f.geometry.coordinates[0];
                  const center = ring[0];
                  const [cx, cy] = project(center[0], center[1]);
                  const r = Math.max(30, f.properties.area_acres * 6);
                  const color = getColor(f.properties.risk_class).fill;
                  return (
                    <circle
                      key={`heat-${f.properties.zone_id}`}
                      cx={cx}
                      cy={cy}
                      r={r}
                      fill={color}
                      opacity={f.properties.risk_score * 0.85}
                    />
                  );
                })}
            </g>
          )}

          {/* Contract C4 Risk Zones Layer */}
          {visibleFeatures.map((f) => {
            const isSelected = selectedZone?.properties.zone_id === f.properties.zone_id;
            const isHovered = hoveredZone?.properties.zone_id === f.properties.zone_id;
            const color = getColor(f.properties.risk_class);
            const ring = f.geometry.coordinates[0];

            const pathData = ring
              .map(([lng, lat], i) => {
                const [px, py] = project(lng, lat);
                // 3D pseudo-extrusion offset based on risk_score
                const extY = py - (f.properties.risk_score * 8);
                return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${extY.toFixed(1)}`;
              })
              .join(' ') + ' Z';

            // Center for label placement
            const center = ring[0];
            const [cx, cy] = project(center[0], center[1]);

            return (
              <g
                key={f.properties.zone_id}
                filter="url(#zone-shadow)"
                className="cursor-pointer transition-all duration-200"
                onClick={() => onSelectZone(f)}
                onMouseEnter={() => setHoveredZone(f)}
                onMouseLeave={() => setHoveredZone(null)}
              >
                {/* 3D Extrusion Wall Base */}
                <path
                  d={pathData}
                  fill={color.fill}
                  fillOpacity={isSelected ? 0.85 : isHovered ? 0.75 : 0.55}
                  stroke={isSelected ? '#ffffff' : color.stroke}
                  strokeWidth={isSelected ? 2.5 : isHovered ? 2 : 1.2}
                />

                {/* Zone Label Marker */}
                <circle
                  cx={cx}
                  cy={cy - f.properties.risk_score * 8}
                  r={isSelected ? 6 : 4}
                  fill={color.stroke}
                  stroke="#ffffff"
                  strokeWidth="1.5"
                />

                <text
                  x={cx + 8}
                  y={cy - f.properties.risk_score * 8 + 4}
                  fill="#ffffff"
                  fontSize="10"
                  fontWeight="600"
                  fontFamily="monospace"
                  className="pointer-events-none drop-shadow-md"
                >
                  {f.properties.zone_id} ({f.properties.area_acres} ac)
                </text>
              </g>
            );
          })}
        </svg>

        {/* Hover Floating Tooltip */}
        {hoveredZone && (
          <div className="absolute bottom-6 left-6 z-30 bg-slate-900/95 backdrop-blur-md border border-slate-700/80 p-3 rounded-xl shadow-2xl text-xs space-y-1 pointer-events-none animate-in fade-in">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono font-bold text-white">
                {hoveredZone.properties.zone_id}
              </span>
              <span className={`font-semibold ${getColor(hoveredZone.properties.risk_class).text}`}>
                {hoveredZone.properties.risk_class_name.toUpperCase()}
              </span>
            </div>
            <div className="text-slate-300 flex items-center justify-between gap-4 font-mono text-[11px]">
              <span>Risk: {(hoveredZone.properties.risk_score * 100).toFixed(0)}%</span>
              <span>Acreage: {hoveredZone.properties.area_acres} ac</span>
            </div>
            <div className="text-emerald-400 font-medium text-[11px] pt-0.5">
              Onset Forecast: {hoveredZone.properties.days_to_onset.toFixed(1)} days before visible symptoms
            </div>
            <div className="text-slate-400 text-[10px]">
              Indicator: {hoveredZone.properties.dominant_indicator}
            </div>
          </div>
        )}
      </div>

      {/* Map Bottom Left: Map Legend */}
      <div className="absolute bottom-4 left-4 z-20 flex items-center gap-3 bg-slate-900/90 backdrop-blur-md border border-slate-800 px-3.5 py-2 rounded-xl shadow-lg text-[11px]">
        <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
          Risk Classes:
        </span>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm" />
          <span className="text-slate-300">0 Healthy</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm" />
          <span className="text-slate-300">1 Early Stress</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shadow-sm" />
          <span className="text-slate-300">2 High Blight</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-sm" />
          <span className="text-slate-300">3 Outbreak</span>
        </div>
      </div>

      {/* Zoom / Reset Controls (Bottom Right) */}
      <div className="absolute bottom-4 right-4 z-20 flex flex-col gap-1.5 bg-slate-900/90 backdrop-blur-md border border-slate-800 p-1.5 rounded-xl shadow-lg">
        <button
          onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
          title="Zoom In"
          className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
          title="Zoom Out"
          className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom(1)}
          title="Reset View"
          className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors text-[10px] font-mono font-bold"
        >
          1:1
        </button>
      </div>
    </div>
  );
};
