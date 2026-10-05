import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  BarChart3, 
  ChevronDown, 
  DollarSign, 
  Layers, 
  MapPin, 
  Radar, 
  RotateCw, 
  ShieldAlert, 
  Sparkles, 
  Wheat 
} from 'lucide-react';
import { Header } from './components/Header';
import { KpiCards } from './components/KpiCards';
import { SatelliteMap } from './components/Map/SatelliteMap';
import { ZoneInspector } from './components/ZoneInspector';
import { SpectralChart } from './components/SpectralChart';
import { ForecastSlider } from './components/ForecastSlider';
import { RoiCalculator } from './components/Economics/RoiCalculator';
import { JobModal } from './components/JobModal';
import { ExportModal } from './components/ExportModal';
import { ZoneTable } from './components/ZoneTable';
import { PixelProbeModal } from './components/PixelProbeModal';
import { ApiService } from './services/api';
import { FieldItem, PixelProbeData, SceneItem, ZoneCollection, ZoneFeature } from './types';
import { CONTRACT_SAMPLE_ZONES, SAMPLE_FIELDS, SAMPLE_SCENES } from './fixtures/mockData';
import { sampleHyperspectralPixel } from './services/pixelProbe';

export const App: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<string>('checking...');
  const [isDemoMode, setIsDemoMode] = useState<boolean>(true);
  const [fields, setFields] = useState<FieldItem[]>(SAMPLE_FIELDS);
  const [scenes, setScenes] = useState<SceneItem[]>(SAMPLE_SCENES);
  const [activeField, setActiveField] = useState<FieldItem>(SAMPLE_FIELDS[0]);
  const [activeJobId, setActiveJobId] = useState<string>('job_c4_live_rehearsal');
  const [zones, setZones] = useState<ZoneCollection>(CONTRACT_SAMPLE_ZONES);
  const [selectedZone, setSelectedZone] = useState<ZoneFeature | null>(null);
  const [daysHorizon, setDaysHorizon] = useState<number>(30);
  const [sidebarTab, setSidebarTab] = useState<'matrix' | 'inspector' | 'economics'>('matrix');
  const [isJobModalOpen, setIsJobModalOpen] = useState<boolean>(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);
  const [probedPixel, setProbedPixel] = useState<PixelProbeData | null>(null);
  const [isProbeModalOpen, setIsProbeModalOpen] = useState<boolean>(false);

  // Initial load & health probe
  useEffect(() => {
    async function init() {
      const health = await ApiService.checkHealth();
      setApiStatus(health.status);
      if (health.status === 'ok' || health.status.includes('healthy')) {
        setIsDemoMode(false);
      }
      const loadedFields = await ApiService.getFields();
      if (loadedFields && loadedFields.length > 0) {
        setFields(loadedFields);
        setActiveField(loadedFields[0]);
      }
      const loadedScenes = await ApiService.getScenes();
      if (loadedScenes && loadedScenes.length > 0) {
        setScenes(loadedScenes);
      }
      const loadedZones = await ApiService.getZones('job_c4_live_rehearsal');
      if (loadedZones && loadedZones.features) {
        setZones(loadedZones);
        // Pre-select first critical zone for inspection
        const critical = loadedZones.features.find((f) => f.properties.risk_class === 2);
        if (critical) setSelectedZone(critical);
      }
    }
    init();
  }, []);

  const handleToggleDemoMode = () => {
    const nextMode = !isDemoMode;
    setIsDemoMode(nextMode);
    ApiService.setDemoMode(nextMode);
  };

  const handleJobComplete = async (jobId: string) => {
    setActiveJobId(jobId);
    const newZones = await ApiService.getZones(jobId);
    setZones(newZones);
    if (newZones.features && newZones.features.length > 0) {
      setSelectedZone(newZones.features[0]);
    }
  };

  const handleFieldChange = (fieldId: string) => {
    const found = fields.find((f) => f.id === fieldId);
    if (found) {
      setActiveField(found);
      // Select appropriate zone for that field
      const fieldZones = zones.features.filter((f) => {
        const ring = f.geometry.coordinates[0];
        return Math.abs(ring[0][0] - found.center[0]) < 0.25;
      });
      if (fieldZones.length > 0) {
        setSelectedZone(fieldZones[0]);
      }
    }
  };

  const handleProbePixel = (probe: PixelProbeData) => {
    setProbedPixel(probe);
    setIsProbeModalOpen(true);
  };

  const handleProbeZone = (zone: ZoneFeature) => {
    const ring = zone.geometry.coordinates[0];
    const centerLng = ring.reduce((sum, pt) => sum + pt[0], 0) / ring.length;
    const centerLat = ring.reduce((sum, pt) => sum + pt[1], 0) / ring.length;
    const probe = sampleHyperspectralPixel(centerLat, centerLng, activeField, zones);
    handleProbePixel(probe);
  };

  // Calculate at-risk acres
  const atRiskAcres = (zones.features || []).reduce(
    (acc, f) => (f.properties.risk_class > 0 ? acc + f.properties.area_acres : acc),
    0
  );

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 select-none">
      {/* Top Aerospace Navigation */}
      <Header
        apiStatus={apiStatus}
        isDemoMode={isDemoMode}
        onToggleDemoMode={handleToggleDemoMode}
        onOpenJobModal={() => setIsJobModalOpen(true)}
        onOpenExportModal={() => setIsExportModalOpen(true)}
        activeJobId={activeJobId}
      />

      {/* KPI Ribbon */}
      <KpiCards zones={zones} totalAcres={activeField.total_acres} />

      {/* Main Tactical Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Map View Area (65%) */}
        <div className="flex-1 relative flex flex-col h-full bg-slate-950">
          {/* Real High-Resolution Satellite Map (with integrated Field Selector) */}
          <SatelliteMap
            zones={zones}
            activeField={activeField}
            fields={fields}
            onFieldChange={handleFieldChange}
            selectedZone={selectedZone}
            onSelectZone={(z) => setSelectedZone(z)}
            daysHorizon={daysHorizon}
            onProbePixel={handleProbePixel}
          />

          {/* Floating Forecast Slider Control */}
          <div className="absolute bottom-14 left-1/2 -translate-x-1/2 z-20 w-[90%] max-w-xl">
            <ForecastSlider
              daysHorizon={daysHorizon}
              onChangeDays={(d) => setDaysHorizon(d)}
            />
          </div>
        </div>

        {/* Right Inspection & Analytics Sidebar (35%) */}
        <div className="w-[450px] border-l border-slate-800/90 bg-slate-950 flex flex-col h-full overflow-hidden shadow-2xl">
          {/* Sidebar Tab Bar */}
          <div className="h-10 border-b border-slate-800 bg-slate-950/90 px-2 flex items-center gap-1">
            <button
              onClick={() => setSidebarTab('matrix')}
              className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                sidebarTab === 'matrix'
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Zone Matrix</span>
            </button>
            <button
              onClick={() => setSidebarTab('inspector')}
              className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                sidebarTab === 'inspector'
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Prescription</span>
            </button>
            <button
              onClick={() => setSidebarTab('economics')}
              className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                sidebarTab === 'economics'
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <DollarSign className="w-3.5 h-3.5" />
              <span>ROI</span>
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto">
            {sidebarTab === 'matrix' ? (
              <ZoneTable
                zones={zones}
                activeField={activeField}
                selectedZone={selectedZone}
                onSelectZone={(z) => setSelectedZone(z)}
                onInspectZone={(z) => {
                  setSelectedZone(z);
                  setSidebarTab('inspector');
                }}
                daysHorizon={daysHorizon}
              />
            ) : sidebarTab === 'inspector' ? (
              <ZoneInspector
                selectedZone={selectedZone}
                onClearSelection={() => setSelectedZone(null)}
                onOpenMatrix={() => setSidebarTab('matrix')}
                onProbeZone={handleProbeZone}
              />
            ) : (
              <div className="p-4 space-y-4">
                <RoiCalculator
                  atRiskAcres={atRiskAcres}
                  totalAcres={activeField.total_acres}
                />
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
                  <h4 className="font-semibold text-slate-200 flex items-center gap-1.5">
                    <BarChart3 className="w-4 h-4 text-emerald-400" />
                    <span>Agronomic Economic Advantage</span>
                  </h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    By forecasting fungal blight <b>18–24 days before visible leaf symptoms</b> with the 3D-CNN + ViT hybrid model, treatments transition from crisis response to preventative micro-dose bio-agents.
                  </p>
                  <ul className="text-[11px] text-slate-300 space-y-1 list-disc list-inside">
                    <li>Eliminates full-field prophylactic pesticide dumping</li>
                    <li>Protects market grading from cosmetic fungal leaf scarring</li>
                    <li>Enables targeted drone variable-rate application (VRA)</li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Hyperspectral Spectral Curve (C1 & P2 Explainability) */}
          <div className="p-3 bg-slate-950/90 border-t border-slate-800">
            <SpectralChart
              dominantIndicator={selectedZone?.properties.dominant_indicator}
            />
          </div>
        </div>
      </div>

      {/* Modals */}
      <JobModal
        isOpen={isJobModalOpen}
        onClose={() => setIsJobModalOpen(false)}
        scenes={scenes}
        fields={fields}
        onJobComplete={handleJobComplete}
      />

      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        zones={zones}
        fieldName={activeField.name}
        cropType={activeField.crop_type}
        totalAcres={activeField.total_acres}
      />

      <PixelProbeModal
        isOpen={isProbeModalOpen}
        onClose={() => setIsProbeModalOpen(false)}
        probeData={probedPixel}
      />
    </div>
  );
};

export default App;
