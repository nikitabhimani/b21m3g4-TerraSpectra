import { FieldItem, JobItem, SceneItem, SpectralPoint, ZoneCollection } from '../types';

export const CONTRACT_SAMPLE_ZONES: ZoneCollection = {
  type: "FeatureCollection",
  job_id: "job_c4_live_rehearsal",
  scene_id: "scn_enmap_punjab_2026",
  generated_at: "2026-09-27T12:00:00Z",
  features: [
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.862313, 30.909842],
            [75.863802, 30.909842],
            [75.863802, 30.911120],
            [75.862313, 30.911120],
            [75.862313, 30.909842]
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
            [75.854112, 30.904432],
            [75.857419, 30.904432],
            [75.857419, 30.907411],
            [75.854112, 30.907411],
            [75.854112, 30.904432]
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
            [75.866440, 30.903264],
            [75.868256, 30.903264],
            [75.868256, 30.904964],
            [75.866440, 30.904964],
            [75.866440, 30.903264]
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
            [75.858000, 30.912000],
            [75.861000, 30.912000],
            [75.861000, 30.914500],
            [75.858000, 30.914500],
            [75.858000, 30.912000]
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
    }
  ]
};

export const SAMPLE_FIELDS: FieldItem[] = [
  {
    id: "fld_punjab_corn_01",
    name: "Ludhiana Sector A — Hybrid Maize",
    crop_type: "Maize (Corn)",
    total_acres: 1000.0,
    center: [75.8600, 30.9080],
    boundary: [
      [
        [75.8500, 30.9000],
        [75.8720, 30.9000],
        [75.8720, 30.9160],
        [75.8500, 30.9160],
        [75.8500, 30.9000]
      ]
    ]
  },
  {
    id: "fld_salinas_veg_02",
    name: "Salinas Valley Test Farm — Lettuce/Broccoli",
    crop_type: "Vegetable Crops",
    total_acres: 450.0,
    center: [-121.6555, 36.6777],
    boundary: [
      [
        [-121.6650, 36.6700],
        [-121.6450, 36.6700],
        [-121.6450, 36.6850],
        [-121.6650, 36.6850],
        [-121.6650, 36.6700]
      ]
    ]
  }
];

export const SAMPLE_SCENES: SceneItem[] = [
  {
    scene_id: "scn_enmap_punjab_2026",
    name: "EnMAP L2A — 200 Canonical Bands (Punjab)",
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
    name: "NASA EO-1 Hyperion Benchmark (Indian Pines)",
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

// Generate realistic 200-band spectral profile (400 nm to 2500 nm)
export const SAMPLE_SPECTRAL_PROFILE: SpectralPoint[] = (() => {
  const points: SpectralPoint[] = [];
  const startWl = 400;
  const endWl = 2500;
  const steps = 60; // 60 sample points for high fidelity SVG rendering
  const stepSize = (endWl - startWl) / steps;

  for (let i = 0; i <= steps; i++) {
    const wl = Math.round(startWl + i * stepSize);
    
    // Healthy canopy: low in blue/red, green peak at 550, red edge jump at 680-750, high NIR, water troughs at 1450, 1940
    let healthy = 0.05;
    if (wl < 500) healthy = 0.04;
    else if (wl < 600) healthy = 0.08 + 0.04 * Math.sin(((wl - 500) / 100) * Math.PI); // Green bump
    else if (wl < 680) healthy = 0.04; // Chlorophyll absorption
    else if (wl < 760) healthy = 0.05 + 0.45 * ((wl - 680) / 80); // Red edge steep rise
    else if (wl < 1300) healthy = 0.50 - 0.05 * ((wl - 760) / 540); // NIR plateau
    else if (wl < 1450) healthy = 0.45 - 0.25 * ((wl - 1300) / 150); // Water dip 1
    else if (wl < 1800) healthy = 0.35 + 0.05 * Math.sin(((wl - 1450) / 350) * Math.PI);
    else if (wl < 2000) healthy = 0.15; // Water dip 2
    else healthy = 0.22 - 0.10 * ((wl - 2000) / 500);

    // Stressed canopy: Red edge shifts left (blue shift), lower NIR plateau, higher visible red reflectance
    let stressed = healthy;
    if (wl >= 520 && wl <= 580) {
      stressed = healthy - 0.025; // Photochemical Reflectance Index (PRI) drop
    } else if (wl >= 650 && wl <= 710) {
      stressed = healthy + 0.05; // Chlorophyll breakdown causes visible red leak
    } else if (wl > 710 && wl <= 1100) {
      stressed = healthy * 0.72; // Cellular structure collapse lowers NIR
    } else if (wl > 1100 && wl <= 1800) {
      stressed = healthy * 1.15; // Water loss increases SWIR reflectance
    }

    // Integrated Gradients band importance peak at key diagnostic bands (537nm, 705nm, 970nm)
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
