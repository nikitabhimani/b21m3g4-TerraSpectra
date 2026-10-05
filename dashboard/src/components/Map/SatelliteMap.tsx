import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { 
  Crosshair, 
  Eye, 
  Layers, 
  Maximize2, 
  Plane, 
  Radio, 
  Scan, 
  Sliders, 
  SunMedium, 
  Target, 
  Wheat, 
  Zap 
} from 'lucide-react';
import { FieldItem, PixelProbeData, ZoneCollection, ZoneFeature } from '../../types';
import { sampleHyperspectralPixel } from '../../services/pixelProbe';

interface SatelliteMapProps {
  zones: ZoneCollection;
  activeField: FieldItem;
  fields: FieldItem[];
  onFieldChange: (fieldId: string) => void;
  selectedZone: ZoneFeature | null;
  onSelectZone: (zone: ZoneFeature) => void;
  daysHorizon: number;
  onProbePixel?: (probe: PixelProbeData) => void;
}

export const SatelliteMap: React.FC<SatelliteMapProps> = ({
  zones,
  activeField,
  fields,
  onFieldChange,
  selectedZone,
  onSelectZone,
  daysHorizon,
  onProbePixel,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const zonesLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const heatmapLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const droneLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const probeLayerGroupRef = useRef<L.LayerGroup | null>(null);

  const [isProbeMode, setIsProbeMode] = useState<boolean>(false);
  const isProbeModeRef = useRef<boolean>(false);
  isProbeModeRef.current = isProbeMode;

  const activeFieldRef = useRef(activeField);
  activeFieldRef.current = activeField;
  const zonesRef = useRef(zones);
  zonesRef.current = zones;
  const onProbePixelRef = useRef(onProbePixel);
  onProbePixelRef.current = onProbePixel;

  const [basemapMode, setBasemapMode] = useState<'satellite' | 'dark' | 'topo'>('satellite');
  const [spectralFilter, setSpectralFilter] = useState<'true_color' | 'cir' | 'chlorophyll'>('true_color');
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);
  const [showDronePath, setShowDronePath] = useState<boolean>(false);
  const [isScanning, setIsScanning] = useState<boolean>(true);
  const [cursorCoords, setCursorCoords] = useState<{ lat: string; lng: string } | null>(null);

  const features = zones.features || [];

  // Filter features based on daysHorizon and field boundary intersection
  const visibleFeatures = features.filter((f) => {
    const p = f.properties;
    const ring = f.geometry.coordinates[0];
    const isInsideField =
      Math.abs(ring[0][0] - activeField.center[0]) < 0.25 &&
      Math.abs(ring[0][1] - activeField.center[1]) < 0.25;

    if (!isInsideField) return false;
    return p.risk_class === 0 || p.days_to_onset >= (30 - daysHorizon);
  });

  const getRiskColor = (riskClass: number) => {
    switch (riskClass) {
      case 0:
        return { fill: '#10b981', border: '#34d399', text: 'text-emerald-400' };
      case 1:
        return { fill: '#f59e0b', border: '#fbbf24', text: 'text-amber-400' };
      case 2:
        return { fill: '#f97316', border: '#fb923c', text: 'text-orange-400' };
      case 3:
        return { fill: '#ef4444', border: '#f87171', text: 'text-rose-400' };
      default:
        return { fill: '#64748b', border: '#94a3b8', text: 'text-slate-400' };
    }
  };

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [activeField.center[1], activeField.center[0]],
        zoom: 15,
        zoomControl: false,
        attributionControl: true,
      });

      L.control.zoom({ position: 'bottomright' }).addTo(map);

      const zonesGroup = L.layerGroup().addTo(map);
      const heatmapGroup = L.layerGroup().addTo(map);
      const droneGroup = L.layerGroup().addTo(map);
      const probeGroup = L.layerGroup().addTo(map);

      zonesLayerGroupRef.current = zonesGroup;
      heatmapLayerGroupRef.current = heatmapGroup;
      droneLayerGroupRef.current = droneGroup;
      probeLayerGroupRef.current = probeGroup;

      map.on('mousemove', (e: L.LeafletMouseEvent) => {
        setCursorCoords({
          lat: e.latlng.lat.toFixed(6),
          lng: e.latlng.lng.toFixed(6),
        });
      });

      map.on('click', (e: L.LeafletMouseEvent) => {
        if (isProbeModeRef.current) {
          const lat = e.latlng.lat;
          const lng = e.latlng.lng;
          renderProbeReticle(lat, lng);
          const probe = sampleHyperspectralPixel(lat, lng, activeFieldRef.current, zonesRef.current);
          onProbePixelRef.current?.(probe);
        }
      });

      mapInstanceRef.current = map;
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  const renderProbeReticle = (lat: number, lng: number) => {
    if (!probeLayerGroupRef.current) return;
    probeLayerGroupRef.current.clearLayers();
    const reticleHtml = `
      <div style="position: relative; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; pointer-events: none;">
        <div style="position: absolute; width: 34px; height: 34px; border-radius: 50%; border: 2px solid #06b6d4; opacity: 0.8; animation: ping 1.4s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
        <div style="position: absolute; width: 14px; height: 14px; border-radius: 50%; background: rgba(6, 182, 212, 0.4); border: 2px solid #ffffff; box-shadow: 0 0 10px #06b6d4;"></div>
        <div style="width: 4px; height: 4px; border-radius: 50%; background: #ffffff;"></div>
        <div style="position: absolute; width: 38px; height: 1.5px; background: rgba(6, 182, 212, 0.9);"></div>
        <div style="position: absolute; width: 1.5px; height: 38px; background: rgba(6, 182, 212, 0.9);"></div>
      </div>
    `;
    const reticleIcon = L.divIcon({
      className: 'pixel-probe-hud-reticle',
      html: reticleHtml,
      iconSize: [34, 34],
      iconAnchor: [17, 17],
    });
    L.marker([lat, lng], { icon: reticleIcon }).addTo(probeLayerGroupRef.current);
  };

  // Synchronize map cursor when probe mode toggles
  useEffect(() => {
    if (mapInstanceRef.current) {
      const container = mapInstanceRef.current.getContainer();
      if (isProbeMode) {
        container.style.cursor = 'crosshair';
      } else {
        container.style.cursor = '';
      }
    }
  }, [isProbeMode]);

  // Update Basemap Tiles
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    let url = '';
    let attribution = '';

    if (basemapMode === 'satellite') {
      url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
      attribution = '&copy; Esri &bull; Earthstar Geographics &bull; TerraSpectra';
    } else if (basemapMode === 'dark') {
      url = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
      attribution = '&copy; OpenStreetMap contributors &copy; CARTO';
    } else {
      url = 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png';
      attribution = '&copy; OpenStreetMap contributors &bull; OpenTopoMap';
    }

    const tileLayer = L.tileLayer(url, {
      maxZoom: 19,
      attribution,
    }).addTo(map);

    tileLayerRef.current = tileLayer;
  }, [basemapMode]);

  // Center on active field change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.flyTo([activeField.center[1], activeField.center[0]], 15, {
      duration: 1.2,
    });
  }, [activeField]);

  // Render Polygons, Farm Boundary & Heatmap
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !zonesLayerGroupRef.current || !heatmapLayerGroupRef.current) return;

    zonesLayerGroupRef.current.clearLayers();
    heatmapLayerGroupRef.current.clearLayers();

    // 1. Draw Field Boundary (Dashed Emerald Outline)
    const fieldLatLngs = activeField.boundary[0].map(([lng, lat]) => [lat, lng] as [number, number]);
    L.polygon(fieldLatLngs, {
      color: '#10b981',
      weight: 2,
      dashArray: '6, 6',
      fillColor: '#064e3b',
      fillOpacity: 0.12,
    })
      .bindTooltip(`<b>${activeField.name}</b><br/>${activeField.crop_type} (${activeField.total_acres} ac)`, {
        permanent: false,
        direction: 'center',
      })
      .addTo(zonesLayerGroupRef.current);

    // 2. Draw Contract C4 Risk Zones
    visibleFeatures.forEach((f) => {
      const p = f.properties;
      const isSelected = selectedZone?.properties.zone_id === p.zone_id;
      const colors = getRiskColor(p.risk_class);

      const latLngs = f.geometry.coordinates[0].map(([lng, lat]) => [lat, lng] as [number, number]);

      const poly = L.polygon(latLngs, {
        color: isSelected ? '#ffffff' : colors.border,
        weight: isSelected ? 3.5 : 2,
        fillColor: colors.fill,
        fillOpacity: isSelected ? 0.8 : p.risk_class === 0 ? 0.3 : 0.65,
      });

      poly.on('click', (e: L.LeafletMouseEvent) => {
        if (isProbeModeRef.current) {
          const lat = e.latlng.lat;
          const lng = e.latlng.lng;
          renderProbeReticle(lat, lng);
          const probe = sampleHyperspectralPixel(lat, lng, activeFieldRef.current, zonesRef.current);
          onProbePixelRef.current?.(probe);
        } else {
          onSelectZone(f);
        }
      });

      poly.bindTooltip(
        `
        <div style="font-family: inherit; font-size: 11px; line-height: 1.4; padding: 2px;">
          <div style="font-weight: 700; color: #ffffff; display: flex; justify-content: space-between;">
            <span>ZONE ${p.zone_id.toUpperCase()}</span>
            <span style="color: ${colors.border};">${p.risk_class_name.toUpperCase()}</span>
          </div>
          <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">
            Risk: ${(p.risk_score * 100).toFixed(1)}% &bull; Area: ${p.area_acres} ac
          </div>
          <div style="color: #10b981; font-weight: 600; margin-top: 3px;">
            Onset: ~${p.days_to_onset.toFixed(1)} days before visible symptoms
          </div>
          <div style="color: #cbd5e1; font-size: 10px; margin-top: 2px;">
            Indicator: <i>${p.dominant_indicator}</i>
          </div>
        </div>
        `,
        { sticky: true, opacity: 0.95 }
      );

      poly.addTo(zonesLayerGroupRef.current!);

      // Heatmap Simulation Glow Circle
      if (showHeatmap && p.risk_class > 0) {
        const center = latLngs[0];
        const circle = L.circle(center, {
          radius: Math.max(50, p.area_acres * 14),
          color: colors.fill,
          fillColor: colors.fill,
          fillOpacity: heatmapOpacity * 0.5,
          weight: 0,
        });
        circle.addTo(heatmapLayerGroupRef.current!);
      }
    });
  }, [visibleFeatures, selectedZone, showHeatmap, heatmapOpacity, activeField]);

  // Render Drone Flightpath
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !droneLayerGroupRef.current) return;

    droneLayerGroupRef.current.clearLayers();

    if (!showDronePath) return;

    const riskZones = visibleFeatures.filter((f) => f.properties.risk_class >= 1);
    if (riskZones.length === 0) return;

    const waypoints: [number, number][] = riskZones.map((f) => {
      const ring = f.geometry.coordinates[0];
      return [ring[0][1], ring[0][0]];
    });

    const homeBase: [number, number] = [
      activeField.boundary[0][0][1],
      activeField.boundary[0][0][0],
    ];
    const fullPath = [homeBase, ...waypoints, homeBase];

    L.polyline(fullPath, {
      color: '#06b6d4',
      weight: 2.5,
      dashArray: '8, 8',
      opacity: 0.85,
    }).addTo(droneLayerGroupRef.current);

    waypoints.forEach((pt, i) => {
      L.circleMarker(pt, {
        radius: 6,
        color: '#06b6d4',
        fillColor: '#ffffff',
        fillOpacity: 0.9,
        weight: 2,
      })
        .bindTooltip(`<b>WP-0${i + 1}</b><br/>Targeted VRA Bio-Fungicide Injection`, {
          permanent: false,
        })
        .addTo(droneLayerGroupRef.current!);
    });
  }, [showDronePath, visibleFeatures, activeField]);

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden flex flex-col select-none">
      {/* Top Aerospace Telemetry Ticker (Tactical HUD) */}
      <div className="absolute top-3 left-4 right-4 z-20 flex items-center justify-between pointer-events-none gap-3">
        {/* Left: Farm Field Selector */}
        <div className="bg-slate-950/90 backdrop-blur-md border border-slate-800/90 px-3 py-1.5 rounded-xl shadow-xl flex items-center gap-2 pointer-events-auto">
          <Wheat className="w-4 h-4 text-emerald-400" />
          <select
            value={activeField.id}
            onChange={(e) => onFieldChange(e.target.value)}
            className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer pr-1"
          >
            {fields.map((f) => (
              <option key={f.id} value={f.id} className="bg-slate-900 text-slate-200">
                {f.name} ({f.crop_type})
              </option>
            ))}
          </select>
        </div>

        {/* Center: Mission Ticker */}
        <div className="hidden xl:flex items-center gap-3 bg-slate-950/90 backdrop-blur-md border border-slate-800/90 px-3.5 py-1.5 rounded-xl shadow-xl text-slate-300 font-mono text-[11px] pointer-events-auto">
          <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
            <Radio className="w-3.5 h-3.5 animate-pulse text-emerald-400" />
            <span>SAT: ENMAP-L2A</span>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">
            SUN-EL: <span className="text-white font-medium">54.2°</span>
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">
            BANDS: <span className="text-emerald-400 font-semibold">200 [400–2500nm]</span>
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">
            GSD: <span className="text-white font-medium">30m/px</span>
          </span>
        </div>

        {/* Right Tactical Map Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          {/* Spectral Filter / Vigor Layer */}
          <div className="bg-slate-950/90 backdrop-blur-md border border-slate-800/90 rounded-xl p-1 shadow-xl flex items-center text-xs">
            <button
              onClick={() => setSpectralFilter('true_color')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                spectralFilter === 'true_color'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              RGB Natural
            </button>
            <button
              onClick={() => setSpectralFilter('cir')}
              title="Color-Infrared (CIR 840/670/560nm) - Highlights vegetative vigor"
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                spectralFilter === 'cir'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              CIR Infrared
            </button>
            <button
              onClick={() => setSpectralFilter('chlorophyll')}
              title="Red-Edge Blue Shift (705nm) Pre-Visual Stress Contrast"
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                spectralFilter === 'chlorophyll'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Red-Edge 705nm
            </button>
          </div>

          {/* Basemap Switcher */}
          <div className="bg-slate-950/90 backdrop-blur-md border border-slate-800/90 rounded-xl p-1 shadow-xl flex items-center text-xs">
            <button
              onClick={() => setBasemapMode('satellite')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                basemapMode === 'satellite'
                  ? 'bg-slate-800 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Satellite
            </button>
            <button
              onClick={() => setBasemapMode('dark')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                basemapMode === 'dark'
                  ? 'bg-slate-800 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Dark Grid
            </button>
            <button
              onClick={() => setBasemapMode('topo')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                basemapMode === 'topo'
                  ? 'bg-slate-800 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Topo 3D
            </button>
          </div>

          {/* Hyperspectral Pixel Probe Mode Button */}
          <div className="bg-slate-950/90 backdrop-blur-md border border-slate-800/90 rounded-xl p-1 shadow-xl flex items-center text-xs">
            <button
              onClick={() => setIsProbeMode(!isProbeMode)}
              title="Sample 200-Band Hyperspectral Profile at any coordinate"
              className={`px-2.5 py-1 rounded-lg font-semibold flex items-center gap-1.5 transition-all ${
                isProbeMode
                  ? 'bg-cyan-500 text-slate-950 shadow-[0_0_12px_rgba(6,182,212,0.6)] font-bold'
                  : 'text-cyan-400 hover:text-cyan-200 hover:bg-slate-800/80'
              }`}
            >
              <Target className={`w-3.5 h-3.5 ${isProbeMode ? 'animate-spin-slow text-slate-950' : 'text-cyan-400'}`} />
              <span>{isProbeMode ? 'Probe Active' : 'Probe Pixel'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Real High-Resolution Satellite Map Container */}
      <div 
        ref={mapContainerRef} 
        className={`w-full h-full relative z-10 ${
          spectralFilter === 'cir'
            ? 'filter hue-rotate-[290deg] saturate-150 contrast-125'
            : spectralFilter === 'chlorophyll'
            ? 'filter saturate-200 contrast-110'
            : ''
        }`}
      />

      {/* Floating Active Probe Banner */}
      {isProbeMode && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-30 bg-slate-950/95 border border-cyan-500/70 backdrop-blur-md px-4 py-2 rounded-full shadow-[0_0_24px_rgba(6,182,212,0.4)] text-cyan-200 text-xs font-mono flex items-center gap-3 animate-pulse">
          <Target className="w-4 h-4 text-cyan-400 animate-spin-slow" />
          <span>PROBE ACTIVE: Click anywhere on the map to sample 200-band spectral profile</span>
          <button
            onClick={() => setIsProbeMode(false)}
            className="ml-2 px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 hover:text-white border border-cyan-800 text-[10px] font-sans uppercase font-bold"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Orbital Pushbroom Laser Scan Beam Animation */}
      {isScanning && (
        <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
          <div className="w-full h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_15px_#10b981] opacity-75 animate-radar-sweep" />
        </div>
      )}

      {/* Floating Tactical Overlay Controls (Cleanly positioned on left without overlap) */}
      <div className="absolute top-16 left-4 z-20 flex flex-col gap-2">
        {/* Heatmap Layer Toggle */}
        <div className="flex items-center gap-2 bg-slate-950/85 backdrop-blur-md border border-slate-800 px-3 py-1.5 rounded-xl shadow-lg text-xs">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-1.5 font-medium transition-colors ${
              showHeatmap ? 'text-emerald-400' : 'text-slate-500'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Heatmap Overlay</span>
          </button>
          {showHeatmap && (
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-16 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              title="Heatmap Opacity"
            />
          )}
        </div>

        {/* Drone Spray Flightpath Toggle */}
        <button
          onClick={() => setShowDronePath(!showDronePath)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border shadow-lg backdrop-blur-md transition-all ${
            showDronePath
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
              : 'bg-slate-950/85 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Plane className="w-3.5 h-3.5 text-cyan-400" />
          <span>Drone Spray Mission (DJI T40)</span>
        </button>

        {/* Satellite Radar Sweep Toggle */}
        <button
          onClick={() => setIsScanning(!isScanning)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border shadow-lg backdrop-blur-md transition-all ${
            isScanning
              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
              : 'bg-slate-950/85 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Scan className="w-3.5 h-3.5 text-emerald-400" />
          <span>Orbital Pushbroom Sensor Sweep</span>
        </button>
      </div>

      {/* Bottom Left: Tactical Reticle Coordinates & Graticule */}
      <div className="absolute bottom-4 left-4 z-20 flex items-center gap-3 bg-slate-950/90 backdrop-blur-md border border-slate-800 px-3.5 py-2 rounded-xl shadow-xl text-[11px] font-mono">
        <div className="flex items-center gap-1.5 text-slate-300">
          <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
          <span>
            {cursorCoords
              ? `${cursorCoords.lat}°N, ${cursorCoords.lng}°E`
              : `${activeField.center[1].toFixed(4)}°N, ${activeField.center[0].toFixed(4)}°E`}
          </span>
        </div>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">ELEV: 214m MSL</span>
        <span className="text-slate-700">|</span>
        <span className="text-emerald-400 font-semibold">UTM 43N</span>
      </div>

      {/* Bottom Center: Contract C4 Legend */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 hidden md:flex items-center gap-3.5 bg-slate-950/90 backdrop-blur-md border border-slate-800 px-4 py-2 rounded-xl shadow-xl text-[11px]">
        <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
          Risk Classes (C4):
        </span>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
          <span className="text-slate-300">0 Healthy</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-[0_0_8px_#f59e0b]" />
          <span className="text-slate-300 font-medium">1 Early Stress (Pre-Visual)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shadow-[0_0_8px_#f97316]" />
          <span className="text-slate-300 font-medium">2 High Blight</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_#ef4444]" />
          <span className="text-slate-300 font-medium">3 Outbreak</span>
        </div>
      </div>
    </div>
  );
};
