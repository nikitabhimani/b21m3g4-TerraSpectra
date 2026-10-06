export type RiskClass = 0 | 1 | 2 | 3;

export type RiskClassName = 'healthy' | 'early_stress' | 'high_blight_risk' | 'visible_disease';

export type DominantIndicator = 
  | 'red_edge_shift' 
  | 'pri_decline' 
  | 'chlorophyll_loss' 
  | 'water_stress'
  | 'baseline';

export interface ZoneProperties {
  zone_id: string;
  risk_class: RiskClass;
  risk_class_name: RiskClassName;
  risk_score: number; // 0.0 to 1.0
  area_acres: number;
  days_to_onset: number; // 0.0 to 30.0 days
  dominant_indicator: DominantIndicator;
  recommended_action: string;
}

export interface ZoneFeature {
  type: 'Feature';
  geometry: {
    type: 'Polygon';
    coordinates: number[][][]; // GeoJSON [lng, lat]
  };
  properties: ZoneProperties;
}

export interface ZoneCollection {
  type: 'FeatureCollection';
  job_id: string;
  scene_id: string;
  generated_at: string;
  features: ZoneFeature[];
}

export interface FieldItem {
  id: string;
  name: string;
  crop_type: string;
  total_acres: number;
  center: [number, number]; // [lng, lat]
  boundary: number[][][];
}

export interface SceneItem {
  scene_id: string;
  name: string;
  satellite: 'EnMAP' | 'PRISMA' | 'Hyperion' | 'Synthetic';
  bands: number;
  acquisition_date: string;
  crs: string;
  width: number;
  height: number;
}

export interface JobSummary {
  acres_analyzed: number;
  acres_at_risk: number;
  zones_by_class: {
    healthy: number;
    early_stress: number;
    high_blight_risk: number;
    visible_disease: number;
  };
}

export interface JobItem {
  job_id: string;
  scene_id: string;
  field_id?: string | null;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  progress: number;
  created_at: string;
  updated_at: string;
  error?: string | null;
  summary?: JobSummary | null;
}

export interface JobEvent {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  progress: number;
  step?: string;
  summary?: JobSummary | null;
  error?: string | null;
  timestamp?: string;
}

export interface SpectralPoint {
  wavelength: number;
  healthy: number;
  stressed: number;
  importance: number;
}

export interface PixelIndices {
  ndvi: number;
  ndre: number;
  pri: number;
  rep: number;
  ndwi: number;
  mcari: number;
}

export interface PixelSpectrumBand {
  wavelength: number;
  reflectance: number;
  healthyReflectance: number;
  stressedReflectance: number;
  soilReflectance: number;
  bandIndex: number;
  snr: number;
}

export interface PixelProbeData {
  lat: number;
  lng: number;
  timestamp: string;
  field_id: string;
  field_name: string;
  risk_class: RiskClass;
  risk_class_name: RiskClassName;
  risk_score: number;
  days_to_onset: number;
  dominant_indicator: DominantIndicator;
  indices: PixelIndices;
  spectrum: PixelSpectrumBand[];
  nearestZoneId?: string;
  zone_name?: string;
  diagnosticNotes: string;
  biophysicalMechanisms: string[];
  recommendedIntervention: string;
  elevationMeters: number;
  gsdMeters: number;
  sensorId: string;
}
