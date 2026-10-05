import { 
  DominantIndicator, 
  FieldItem, 
  PixelIndices, 
  PixelProbeData, 
  PixelSpectrumBand, 
  RiskClass, 
  RiskClassName, 
  ZoneCollection, 
  ZoneFeature 
} from '../types';

// Canonical 200-band wavelength grid (Contract C1: 400nm to 2500nm, ~10.55nm FWHM)
export const CANONICAL_WAVELENGTHS: number[] = Array.from({ length: 200 }, (_, i) => 
  Number((400 + i * 10.55276).toFixed(2))
);

/**
 * Standard ray-casting point-in-polygon algorithm.
 * Coordinates are [lng, lat].
 */
export function isPointInPolygon(point: [number, number], polygon: number[][]): boolean {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * Calculate distance (in degrees approx) between two coordinates [lng, lat].
 */
function getCoordDistance(c1: [number, number], c2: [number, number]): number {
  const dx = c1[0] - c2[0];
  const dy = c1[1] - c2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Find reflectance at closest available band to target wavelength.
 */
function getReflectanceAt(spectrum: { wavelength: number; reflectance: number }[], targetWl: number): number {
  let closest = spectrum[0];
  let minDiff = Math.abs(spectrum[0].wavelength - targetWl);

  for (let i = 1; i < spectrum.length; i++) {
    const diff = Math.abs(spectrum[i].wavelength - targetWl);
    if (diff < minDiff) {
      minDiff = diff;
      closest = spectrum[i];
    }
  }

  return closest.reflectance;
}

/**
 * Guyot & Baret (1988) 4-point linear interpolation model for Red-Edge Position (REP).
 * Estimates the inflection point wavelength (nm) between 690nm and 740nm.
 */
function calculateREP(
  r670: number, 
  r700: number, 
  r740: number, 
  r780: number
): number {
  const rRe = (r670 + r780) / 2;
  const denom = r740 - r700;
  if (Math.abs(denom) < 1e-5) return 715.0;

  const rep = 700 + 40 * ((rRe - r700) / denom);
  return Number(Math.max(692, Math.min(738, rep)).toFixed(1));
}

/**
 * Generate synthetic physics-grounded reflectance across the 200-band spectrum
 * based on vegetation health state and PROSAIL radiative transfer parameters.
 */
export function generateBandReflectance(
  wavelength: number,
  riskClass: RiskClass,
  isSoil: boolean = false
): { reflectance: number; healthy: number; stressed: number; soil: number } {
  const wl = wavelength;

  // 1. Healthy reference canopy (high chlorophyll a/b, dense mesophyll cells)
  let healthy = 0.04;
  if (wl < 500) {
    healthy = 0.038 + 0.005 * Math.sin(((wl - 400) / 100) * Math.PI);
  } else if (wl < 600) {
    // Green reflectance peak (550nm)
    healthy = 0.05 + 0.065 * Math.sin(((wl - 500) / 100) * Math.PI);
  } else if (wl < 685) {
    // Chlorophyll absorption pit (670nm)
    healthy = 0.035 + 0.01 * Math.cos(((wl - 600) / 85) * Math.PI);
  } else if (wl < 760) {
    // Steep red-edge rise (685 - 750nm)
    const t = (wl - 685) / 75;
    healthy = 0.045 + 0.46 * (1 / (1 + Math.exp(-10 * (t - 0.5))));
  } else if (wl < 1300) {
    // NIR scattering plateau (760 - 1300nm)
    healthy = 0.51 - 0.06 * ((wl - 760) / 540);
    // 970nm weak water absorption dip
    if (Math.abs(wl - 970) < 50) {
      healthy -= 0.035 * Math.exp(-Math.pow((wl - 970) / 28, 2));
    }
  } else if (wl < 1450) {
    // Transition to water absorption
    healthy = 0.45 - 0.28 * ((wl - 1300) / 150);
  } else if (wl < 1850) {
    // SWIR-1 window
    healthy = 0.32 + 0.07 * Math.sin(((wl - 1450) / 400) * Math.PI);
  } else if (wl < 2000) {
    // Atmospheric water vapor pit
    healthy = 0.12 + 0.04 * ((wl - 1850) / 150);
  } else {
    // SWIR-2 window
    healthy = 0.24 - 0.11 * ((wl - 2000) / 500);
  }

  // 2. Bare soil baseline (monotonically rising with no red-edge step)
  const soil = Math.min(0.48, 0.09 + 0.32 * ((wl - 400) / 2100) + 0.015 * Math.sin((wl / 300) * Math.PI));

  // 3. Modulate for Stressed / Diseased Canopy
  let stressed = healthy;
  // Early pre-visual stress:
  // a) PRI depression (520-575nm)
  if (wl >= 520 && wl <= 575) {
    stressed -= 0.024 * Math.exp(-Math.pow((wl - 535) / 22, 2));
  }
  // b) Red-Edge Blue Shift: absorption at 675-735nm shifts left, lowering inflection point to ~706-710nm
  if (wl >= 675 && wl <= 735) {
    stressed = healthy + 0.085 * Math.sin(((wl - 675) / 60) * Math.PI);
  }
  // c) NIR mesophyll degradation (740-1100nm)
  if (wl > 735 && wl <= 1100) {
    stressed *= 0.74;
  }
  // d) Loss of leaf water content (increased SWIR reflectance)
  if (wl > 1150 && wl <= 1800) {
    stressed = Math.min(0.55, stressed * 1.22);
  }

  // Current probed pixel reflectance determination
  let current = healthy;
  if (isSoil) {
    current = soil;
  } else {
    switch (riskClass) {
      case 0: // Healthy
        current = healthy;
        break;
      case 1: // Early Stress (Pre-visual phase)
        current = healthy * 0.45 + stressed * 0.55;
        break;
      case 2: // High Blight Risk
        current = stressed;
        break;
      case 3: // Visible Disease / Necrotic lesions
        // Severe chlorosis: red pit disappears, NIR collapses
        if (wl < 700) {
          current = healthy + 0.14 * Math.exp(-Math.pow((wl - 670) / 60, 2));
        } else if (wl <= 1200) {
          current = healthy * 0.48;
        } else {
          current = soil * 0.85;
        }
        break;
    }
  }

  // Slight pseudo-noise based on wavelength to mimic real sensor SNR
  const noise = (Math.sin(wl * 13.7) * 0.0025);
  const clamp = (v: number) => Math.max(0.01, Math.min(0.95, Number((v + noise).toFixed(4))));

  return {
    reflectance: clamp(current),
    healthy: clamp(healthy),
    stressed: clamp(stressed),
    soil: clamp(soil),
  };
}

/**
 * Execute pixel probe at specific GPS coordinates [lat, lng].
 */
export function sampleHyperspectralPixel(
  lat: number,
  lng: number,
  activeField: FieldItem,
  zones: ZoneCollection
): PixelProbeData {
  const probeCoord: [number, number] = [lng, lat];
  const fieldBoundary = activeField.boundary[0];
  const isInsideField = isPointInPolygon(probeCoord, fieldBoundary);

  // Check if inside any risk zone
  let targetZone: ZoneFeature | null = null;
  let minDistance = Infinity;
  let nearestZone: ZoneFeature | null = null;

  for (const f of zones.features || []) {
    const ring = f.geometry.coordinates[0];
    if (isPointInPolygon(probeCoord, ring)) {
      targetZone = f;
      break;
    }
    const centerLng = ring.reduce((sum, pt) => sum + pt[0], 0) / ring.length;
    const centerLat = ring.reduce((sum, pt) => sum + pt[1], 0) / ring.length;
    const dist = getCoordDistance(probeCoord, [centerLng, centerLat]);
    if (dist < minDistance) {
      minDistance = dist;
      nearestZone = f;
    }
  }

  let riskClass: RiskClass = 0;
  let riskClassName: RiskClassName = 'healthy';
  let riskScore = 0.06;
  let daysToOnset = 30.0;
  let dominantIndicator: DominantIndicator = 'baseline';
  let nearestZoneId: string | undefined = undefined;
  let zoneName: string | undefined = undefined;
  let isSoil = false;

  if (targetZone) {
    const p = targetZone.properties;
    riskClass = p.risk_class;
    riskClassName = p.risk_class_name;
    riskScore = p.risk_score;
    daysToOnset = p.days_to_onset;
    dominantIndicator = p.dominant_indicator;
    nearestZoneId = p.zone_id;
    zoneName = `Zone ${p.zone_id.replace('zone-', '').toUpperCase()}`;
  } else if (isInsideField) {
    // Inside field parcel but not strictly in a designated risk zone
    // Check if within buffer zone (< 0.005 degrees) of a risk zone
    if (nearestZone && minDistance < 0.004) {
      const p = nearestZone.properties;
      riskClass = 1;
      riskClassName = 'early_stress';
      riskScore = Number((p.risk_score * 0.65).toFixed(2));
      daysToOnset = Number((p.days_to_onset + 4.5).toFixed(1));
      dominantIndicator = p.dominant_indicator;
      nearestZoneId = p.zone_id;
      zoneName = `Zone ${p.zone_id.replace('zone-', '').toUpperCase()} Peripheral Buffer`;
    } else {
      riskClass = 0;
      riskClassName = 'healthy';
      riskScore = 0.08;
      daysToOnset = 30.0;
      dominantIndicator = 'baseline';
      zoneName = `${activeField.name} Baseline Parcel`;
    }
  } else {
    // Outside active field boundary: Soil / Parcel Boundary
    isSoil = true;
    riskClass = 0;
    riskClassName = 'healthy';
    riskScore = 0.02;
    daysToOnset = 30.0;
    dominantIndicator = 'baseline';
    zoneName = 'External Field Boundary (Bare Soil / Fallow)';
  }

  // Generate 200-band full spectrum
  const spectrum: PixelSpectrumBand[] = CANONICAL_WAVELENGTHS.map((wl, idx) => {
    const values = generateBandReflectance(wl, riskClass, isSoil);
    // Sensor SNR calculation (EnMAP VNIR: ~220, SWIR: ~160)
    const baseSnr = wl < 1000 ? 220 : 160;
    const snr = Math.round(baseSnr * (values.reflectance / 0.5) + (Math.sin(idx) * 8));

    return {
      wavelength: wl,
      reflectance: values.reflectance,
      healthyReflectance: values.healthy,
      stressedReflectance: values.stressed,
      soilReflectance: values.soil,
      bandIndex: idx + 1,
      snr: Math.max(80, snr),
    };
  });

  // Calculate Agronomic Spectral Indices
  const r531 = getReflectanceAt(spectrum, 531);
  const r550 = getReflectanceAt(spectrum, 550);
  const r570 = getReflectanceAt(spectrum, 570);
  const r670 = getReflectanceAt(spectrum, 670);
  const r700 = getReflectanceAt(spectrum, 700);
  const r720 = getReflectanceAt(spectrum, 720);
  const r740 = getReflectanceAt(spectrum, 740);
  const r780 = getReflectanceAt(spectrum, 780);
  const r790 = getReflectanceAt(spectrum, 790);
  const r840 = getReflectanceAt(spectrum, 840);
  const r860 = getReflectanceAt(spectrum, 860);
  const r1240 = getReflectanceAt(spectrum, 1240);

  const ndvi = Number(((r840 - r670) / (r840 + r670 || 1e-5)).toFixed(3));
  const ndre = Number(((r790 - r720) / (r790 + r720 || 1e-5)).toFixed(3));
  const pri = Number(((r531 - r570) / (r531 + r570 || 1e-5)).toFixed(3));
  const rep = calculateREP(r670, r700, r740, r780);
  const ndwi = Number(((r860 - r1240) / (r860 + r1240 || 1e-5)).toFixed(3));
  const mcari = Number((((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 || 1e-5))).toFixed(3));

  const indices: PixelIndices = {
    ndvi,
    ndre,
    pri,
    rep,
    ndwi,
    mcari,
  };

  // Compile Biophysical Mechanisms & Explanations
  const biophysicalMechanisms: string[] = [];
  let diagnosticNotes = '';
  let recommendedIntervention = '';

  if (isSoil) {
    diagnosticNotes = 'Sampled pixel falls outside active vegetative canopy. Spectrum matches dry mineral soil / fallow border with flat absorption profile.';
    biophysicalMechanisms.push('Absence of photosynthetic red-edge transition step (690-740nm).');
    biophysicalMechanisms.push('Monotonic albedo increase from visible to shortwave infrared spectrum.');
    recommendedIntervention = 'No agronomic treatment required for boundary parcel.';
  } else if (riskClass === 2) {
    diagnosticNotes = `High-confidence pre-symptomatic blight alert (~${daysToOnset.toFixed(1)} days to symptom expression). Severe 705nm red-edge inflection shift (-16nm blue-shift) with prominent xanthophyll cycle PRI depression (${pri}).`;
    biophysicalMechanisms.push('Early hyphal penetration degrading spongy mesophyll leaf structure prior to visible yellowing.');
    biophysicalMechanisms.push('Excess energy dissipation in photosystem II antenna complexes detected via 531nm reflectance drop.');
    biophysicalMechanisms.push('Cellular dehydration signature evidenced by elevated 1240nm reflectance.');
    recommendedIntervention = 'Schedule targeted Variable-Rate Application (VRA) preventative bio-fungicide (e.g. Serenade ASO / Bacillus amyloliquefaciens at 2.5 L/ha) within 72 hours.';
  } else if (riskClass === 1) {
    diagnosticNotes = `Early physiological stress anomaly detected (~${daysToOnset.toFixed(1)} days before visible symptoms). Photochemical Reflectance Index is depressed (${pri}), indicating sub-visual light-use efficiency drop.`;
    biophysicalMechanisms.push('Elevated non-photochemical quenching (NPQ) signaling latent fungal pathogen colonization.');
    biophysicalMechanisms.push('Minor red-edge inflection contraction to 712nm (chlorophyll degradation in early phase).');
    recommendedIntervention = 'Deploy multispectral high-resolution scouting drone at 15m AGL. Re-image in 4 days to confirm whether stress resolves or advances.';
  } else if (riskClass === 3) {
    diagnosticNotes = 'Active necrotic lesion zone. Major chlorophyll breakdown, red absorption collapse at 670nm, and mesophyll cell wall disintegration.';
    biophysicalMechanisms.push('Complete loss of photosynthetic capacity across the 640-680nm chlorophyll absorption well.');
    biophysicalMechanisms.push('Total breakdown of spongy mesophyll scattering (NIR plateau collapsed below 0.25).');
    recommendedIntervention = 'Emergency curative fungicide barrier application to ring-fence the focal infection and contain airborne spore release.';
  } else {
    diagnosticNotes = 'Canopy in prime photosynthetic health. High chlorophyll content, steep red-edge slope (REP 724.8nm), and optimal water retention.';
    biophysicalMechanisms.push('Optimal photosystem II photochemical efficiency (positive PRI).');
    biophysicalMechanisms.push('Intact bi-facial leaf mesophyll with strong NIR backscattering.');
    recommendedIntervention = 'Maintain routine baseline satellite monitoring. No treatment or intervention necessary.';
  }

  // Ground elevation calculation (approx based on Punjab plain / Po Valley)
  const baseElev = activeField.id.includes('punjab') ? 245 : 68;
  const elevationMeters = Number((baseElev + Math.sin(lat * 100) * 4.2).toFixed(1));

  return {
    lat: Number(lat.toFixed(6)),
    lng: Number(lng.toFixed(6)),
    timestamp: new Date().toISOString(),
    field_id: activeField.id,
    field_name: activeField.name,
    risk_class: riskClass,
    risk_class_name: riskClassName,
    risk_score: riskScore,
    days_to_onset: daysToOnset,
    dominant_indicator: dominantIndicator,
    indices,
    spectrum,
    nearestZoneId,
    zone_name: zoneName,
    diagnosticNotes,
    biophysicalMechanisms,
    recommendedIntervention,
    elevationMeters,
    gsdMeters: 30.0,
    sensorId: 'EnMAP-L2A (200-Band Resampled)',
  };
}

/**
 * Format probed pixel spectrum as CSV string for agronomic laboratory analysis.
 */
export function exportSpectrumToCsv(probe: PixelProbeData): string {
  const headers = [
    'Band_Index',
    'Wavelength_nm',
    'Reflectance_Sampled',
    'Reflectance_Healthy_Ref',
    'Reflectance_Stressed_Ref',
    'Reflectance_Soil_Ref',
    'Sensor_SNR',
  ];

  const rows = probe.spectrum.map((b) => [
    b.bandIndex,
    b.wavelength,
    b.reflectance,
    b.healthyReflectance,
    b.stressedReflectance,
    b.soilReflectance,
    b.snr,
  ].join(','));

  const metadata = [
    `# TerraSpectra Hyperspectral Pixel Probe Report`,
    `# Probed Location: ${probe.lat} N, ${probe.lng} E`,
    `# Field: ${probe.field_name} (${probe.field_id})`,
    `# Sensor: ${probe.sensorId}`,
    `# GSD: ${probe.gsdMeters}m`,
    `# Risk Assessment: ${probe.risk_class_name.toUpperCase()} (Score: ${(probe.risk_score * 100).toFixed(1)}%, Onset: ~${probe.days_to_onset.toFixed(1)} days)`,
    `# Spectral Indices: NDVI=${probe.indices.ndvi}, NDRE=${probe.indices.ndre}, PRI=${probe.indices.pri}, REP=${probe.indices.rep}nm, NDWI=${probe.indices.ndwi}`,
    `# Generated: ${probe.timestamp}`,
    `# --------------------------------------------------`,
  ].join('\n');

  return `${metadata}\n${headers.join(',')}\n${rows.join('\n')}`;
}

/**
 * Format probe telemetry as structured JSON string.
 */
export function exportProbeToJson(probe: PixelProbeData): string {
  return JSON.stringify(probe, null, 2);
}
