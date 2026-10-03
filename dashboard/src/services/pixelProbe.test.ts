import { describe, expect, it } from 'vitest';
import { 
  CANONICAL_WAVELENGTHS, 
  exportProbeToJson, 
  exportSpectrumToCsv, 
  generateBandReflectance, 
  isPointInPolygon, 
  sampleHyperspectralPixel 
} from './pixelProbe';
import { CONTRACT_SAMPLE_ZONES, SAMPLE_FIELDS } from '../fixtures/mockData';

describe('Pixel Probe Service', () => {
  it('has exactly 200 canonical wavelengths starting at 400nm and reaching ~2500nm', () => {
    expect(CANONICAL_WAVELENGTHS).toHaveLength(200);
    expect(CANONICAL_WAVELENGTHS[0]).toBe(400);
    expect(CANONICAL_WAVELENGTHS[199]).toBeGreaterThan(2490);
  });

  describe('isPointInPolygon', () => {
    const square = [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0],
    ];

    it('correctly identifies points inside polygon', () => {
      expect(isPointInPolygon([5, 5], square)).toBe(true);
      expect(isPointInPolygon([1, 1], square)).toBe(true);
    });

    it('correctly identifies points outside polygon', () => {
      expect(isPointInPolygon([15, 5], square)).toBe(false);
      expect(isPointInPolygon([-1, 5], square)).toBe(false);
    });
  });

  describe('generateBandReflectance', () => {
    it('produces typical vegetation red-edge jump for healthy canopy', () => {
      const red = generateBandReflectance(670, 0); // chlorophyll dip
      const nir = generateBandReflectance(840, 0); // NIR plateau
      expect(red.healthy).toBeLessThan(0.10);
      expect(nir.healthy).toBeGreaterThan(0.40);
      expect(nir.healthy).toBeGreaterThan(red.healthy * 4);
    });

    it('demonstrates red-edge blue-shift and PRI depression for stressed canopy', () => {
      const healthyPri = generateBandReflectance(535, 0);
      const stressedPri = generateBandReflectance(535, 2);
      expect(stressedPri.stressed).toBeLessThan(healthyPri.healthy);
    });
  });

  describe('sampleHyperspectralPixel', () => {
    const field = SAMPLE_FIELDS[0]; // Punjab West Farm
    const zones = CONTRACT_SAMPLE_ZONES;

    it('samples high risk blight zone accurately', () => {
      // Coordinates inside zone-red-01 ([75.7220, 30.8520] to [75.7275, 30.8565])
      const lat = 30.8540;
      const lng = 75.7245;

      const probe = sampleHyperspectralPixel(lat, lng, field, zones);

      expect(probe.lat).toBe(lat);
      expect(probe.lng).toBe(lng);
      expect(probe.field_id).toBe(field.id);
      expect(probe.risk_class).toBe(2);
      expect(probe.risk_class_name).toBe('high_blight_risk');
      expect(probe.nearestZoneId).toBe('zone-red-01');
      expect(probe.spectrum).toHaveLength(200);

      // Verify calculated indices
      expect(probe.indices.ndvi).toBeGreaterThan(0.5);
      expect(probe.indices.pri).toBeLessThan(0.0); // depressed PRI
      expect(probe.indices.rep).toBeLessThan(715); // blue shift!
      expect(probe.diagnosticNotes).toContain('pre-symptomatic blight alert');
    });

    it('samples healthy field parcel when far from infected zones', () => {
      // In Punjab field but far from infected zones
      const lat = 30.8420;
      const lng = 75.7150;

      const probe = sampleHyperspectralPixel(lat, lng, field, zones);

      expect(probe.risk_class).toBe(0);
      expect(probe.risk_class_name).toBe('healthy');
      expect(probe.indices.rep).toBeGreaterThan(718); // Normal healthy REP
      expect(probe.biophysicalMechanisms.length).toBeGreaterThan(0);
    });

    it('samples soil when outside active field parcel', () => {
      const lat = 30.9500;
      const lng = 75.9500;

      const probe = sampleHyperspectralPixel(lat, lng, field, zones);

      expect(probe.zone_name).toContain('External Field Boundary');
      expect(probe.diagnosticNotes).toContain('dry mineral soil');
      expect(probe.indices.ndvi).toBeLessThan(0.3); // low NDVI for soil
    });
  });

  describe('Export functions', () => {
    const field = SAMPLE_FIELDS[0];
    const zones = CONTRACT_SAMPLE_ZONES;
    const probe = sampleHyperspectralPixel(30.8540, 75.7245, field, zones);

    it('exports CSV with 200 data rows and metadata headers', () => {
      const csv = exportSpectrumToCsv(probe);
      expect(csv).toContain('# TerraSpectra Hyperspectral Pixel Probe Report');
      expect(csv).toContain('Wavelength_nm,Reflectance_Sampled');
      const lines = csv.trim().split('\n');
      expect(lines.length).toBeGreaterThan(205);
    });

    it('exports structured JSON with complete telemetry payload', () => {
      const jsonStr = exportProbeToJson(probe);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.lat).toBe(30.854);
      expect(parsed.indices.ndvi).toBeDefined();
      expect(parsed.spectrum).toHaveLength(200);
    });
  });
});
