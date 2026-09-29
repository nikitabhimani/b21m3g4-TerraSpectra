import { FieldItem, JobItem, SceneItem, SpectralPoint, ZoneCollection } from '../types';

export const CONTRACT_SAMPLE_ZONES: ZoneCollection = {
  type: "FeatureCollection",
  job_id: "job_c4_live_rehearsal",
  scene_id: "scn_enmap_punjab_2026",
  generated_at: "2026-09-27T12:00:00Z",
  features: [
    // --- Punjab Agricultural Belt (Ludhiana West Farm) ---
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.7220, 30.8520],
            [75.7275, 30.8520],
            [75.7275, 30.8565],
            [75.7220, 30.8565],
            [75.7220, 30.8520]
          ]
        ]
      },
      properties: {
        zone_id: "zone-red-01",
        risk_class: 2,
        risk_class_name: "high_blight_risk",
        risk_score: 0.91,
        area_acres: 5.2,
        days_to_onset: 18.5,
        dominant_indicator: "red_edge_shift",
        recommended_action: "Targeted preventative bio-fungicide spray within 7 days. Inspect field grid B-4."
      }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.7140, 30.8460],
            [75.7210, 30.8460],
            [75.7210, 30.8510],
            [75.7140, 30.8510],
            [75.7140, 30.8460]
          ]
        ]
      },
      properties: {
        zone_id: "zone-amber-02",
        risk_class: 1,
        risk_class_name: "early_stress",
        risk_score: 0.68,
        area_acres: 12.8,
        days_to_onset: 24.0,
        dominant_indicator: "pri_decline",
        recommended_action: "Elevate drone scouting frequency; re-image in 5 days for chlorophyll recovery."
      }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.7280, 30.8440],
            [75.7320, 30.8440],
            [75.7320, 30.8475],
            [75.7280, 30.8475],
            [75.7280, 30.8440]
          ]
        ]
      },
      properties: {
        zone_id: "zone-crit-03",
        risk_class: 3,
        risk_class_name: "visible_disease",
        risk_score: 0.98,
        area_acres: 1.8,
        days_to_onset: 0.0,
        dominant_indicator: "chlorophyll_loss",
        recommended_action: "Immediate localized curative treatment and ground barrier isolation."
      }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.7110, 30.8520],
            [75.7190, 30.8520],
            [75.7190, 30.8580],
            [75.7110, 30.8580],
            [75.7110, 30.8520]
          ]
        ]
      },
      properties: {
        zone_id: "zone-green-04",
        risk_class: 0,
        risk_class_name: "healthy",
        risk_score: 0.08,
        area_acres: 34.5,
        days_to_onset: 30.0,
        dominant_indicator: "baseline",
        recommended_action: "Canopy is vigorous. Maintain standard irrigation and nutrition schedule."
      }
    },

    // --- Salinas Valley Agricultural Parcel ---
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-121.6580, 36.6740],
            [-121.6520, 36.6740],
            [-121.6520, 36.6785],
            [-121.6580, 36.6785],
            [-121.6580, 36.6740]
          ]
        ]
      },
      properties: {
        zone_id: "salinas-red-01",
        risk_class: 2,
        risk_class_name: "high_blight_risk",
        risk_score: 0.94,
        area_acres: 8.4,
        days_to_onset: 16.0,
        dominant_indicator: "red_edge_shift",
        recommended_action: "Apply systemic strobilurin fungicide within 48h to prevent downy mildew outbreak."
      }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-121.6640, 36.6710],
            [-121.6590, 36.6710],
            [-121.6590, 36.6760],
            [-121.6640, 36.6760],
            [-121.6640, 36.6710]
          ]
        ]
      },
      properties: {
        zone_id: "salinas-amber-02",
        risk_class: 1,
        risk_class_name: "early_stress",
        risk_score: 0.72,
        area_acres: 15.2,
        days_to_onset: 22.0,
        dominant_indicator: "pri_decline",
        recommended_action: "Monitor micro-climate humidity; deploy bio-stimulant copper spray."
      }
    }
  ]
};

export const SAMPLE_FIELDS: FieldItem[] = [
  {
    id: "fld_punjab_corn_01",
    name: "Ludhiana Ag Belt — Hybrid Maize",
    crop_type: "Maize (Corn)",
    total_acres: 1000.0,
    center: [75.7220, 30.8520],
    boundary: [
      [
        [75.7050, 30.8400],
        [75.7400, 30.8400],
        [75.7400, 30.8650],
        [75.7050, 30.8650],
        [75.7050, 30.8400]
      ]
    ]
  },
  {
    id: "fld_salinas_veg_02",
    name: "Salinas Valley Benchmark — Lettuce/Broccoli",
    crop_type: "Vegetable Crops",
    total_acres: 450.0,
    center: [-121.6555, 36.6777],
    boundary: [
      [
        [-121.6680, 36.6680],
        [-121.6420, 36.6680],
        [-121.6420, 36.6860],
        [-121.6680, 36.6860],
        [-121.6680, 36.6680]
      ]
    ]
  }
];

export const SAMPLE_SCENES: SceneItem[] = [
  {
    scene_id: "scn_enmap_punjab_2026",
    name: "EnMAP L2A — 200 Canonical Bands (Punjab Ag Belt)",
    satellite: "EnMAP",
    bands: 200,
    acquisition_date: "2026-09-25",
    crs: "EPSG:32643",
    width: 1024,
    height: 1024
  },
  {
    scene_id: "scn_prisma_italy_2026",
    name: "PRISMA Hyperspectral Tile (Po Valley)",
    satellite: "PRISMA",
    bands: 200,
    acquisition_date: "2026-09-22",
    crs: "EPSG:32632",
    width: 1024,
    height: 1024
  },
  {
    scene_id: "scn_hyperion_indiana_benchmark",
    name: "NASA EO-1 Hyperion Benchmark (Salinas Valley)",
    satellite: "Hyperion",
    bands: 200,
    acquisition_date: "2026-09-18",
    crs: "EPSG:32616",
    width: 512,
    height: 512
  }
];

export const SAMPLE_JOB: JobItem = {
  job_id: "job_c4_live_rehearsal",
  scene_id: "scn_enmap_punjab_2026",
  field_id: "fld_punjab_corn_01",
  status: "succeeded",
  progress: 1.0,
  created_at: "2026-09-27T11:58:12Z",
  updated_at: "2026-09-27T12:00:00Z",
  error: null,
  summary: {
    acres_analyzed: 1000.0,
    acres_at_risk: 19.8,
    zones_by_class: {
      healthy: 1,
      early_stress: 1,
      high_blight_risk: 1,
      visible_disease: 1
    }
  }
};

export const SAMPLE_SPECTRAL_PROFILE: SpectralPoint[] = (() => {
  const points: SpectralPoint[] = [];
  const startWl = 400;
  const endWl = 2500;
  const steps = 60;
  const stepSize = (endWl - startWl) / steps;

  for (let i = 0; i <= steps; i++) {
    const wl = Math.round(startWl + i * stepSize);
    
    let healthy = 0.05;
    if (wl < 500) healthy = 0.04;
    else if (wl < 600) healthy = 0.08 + 0.04 * Math.sin(((wl - 500) / 100) * Math.PI);
    else if (wl < 680) healthy = 0.04;
    else if (wl < 760) healthy = 0.05 + 0.45 * ((wl - 680) / 80);
    else if (wl < 1300) healthy = 0.50 - 0.05 * ((wl - 760) / 540);
    else if (wl < 1450) healthy = 0.45 - 0.25 * ((wl - 1300) / 150);
    else if (wl < 1800) healthy = 0.35 + 0.05 * Math.sin(((wl - 1450) / 350) * Math.PI);
    else if (wl < 2000) healthy = 0.15;
    else healthy = 0.22 - 0.10 * ((wl - 2000) / 500);

    let stressed = healthy;
    if (wl >= 520 && wl <= 580) {
      stressed = healthy - 0.025;
    } else if (wl >= 650 && wl <= 710) {
      stressed = healthy + 0.05;
    } else if (wl > 710 && wl <= 1100) {
      stressed = healthy * 0.72;
    } else if (wl > 1100 && wl <= 1800) {
      stressed = healthy * 1.15;
    }

    let importance = 0.002;
    if (Math.abs(wl - 537) < 30) importance = 0.015 * (1 - Math.abs(wl - 537) / 30);
    else if (Math.abs(wl - 705) < 35) importance = 0.018 * (1 - Math.abs(wl - 705) / 35);
    else if (Math.abs(wl - 970) < 40) importance = 0.011 * (1 - Math.abs(wl - 970) / 40);

    points.push({
      wavelength: wl,
      healthy: Math.max(0.01, Math.min(1.0, healthy)),
      stressed: Math.max(0.01, Math.min(1.0, stressed)),
      importance: Number(importance.toFixed(4))
    });
  }
  return points;
})();
