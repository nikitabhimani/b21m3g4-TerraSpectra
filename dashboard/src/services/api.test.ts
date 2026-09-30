import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiService } from './api';
import {
  CONTRACT_SAMPLE_ZONES,
  SAMPLE_FIELDS,
  SAMPLE_SCENES,
} from '../fixtures/mockData';

describe('ApiService Client & Contract C4 Compliance', () => {
  beforeEach(() => {
    ApiService.setDemoMode(false);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    ApiService.setDemoMode(false);
    vi.restoreAllMocks();
  });

  describe('Demo Mode Operations', () => {
    it('manages demo mode state correctly', () => {
      expect(ApiService.getDemoMode()).toBe(false);
      ApiService.setDemoMode(true);
      expect(ApiService.getDemoMode()).toBe(true);
    });

    it('returns healthy demo payload when demo mode is active', async () => {
      ApiService.setDemoMode(true);
      const health = await ApiService.checkHealth();
      expect(health.status).toBe('healthy (demo fixture)');
      expect(health.cuda).toBe(true);
      expect(health.version).toBe('0.1.0');
    });

    it('returns sample scenes and fields in demo mode', async () => {
      ApiService.setDemoMode(true);
      const scenes = await ApiService.getScenes();
      const fields = await ApiService.getFields();

      expect(scenes).toEqual(SAMPLE_SCENES);
      expect(fields).toEqual(SAMPLE_FIELDS);
      expect(scenes.length).toBeGreaterThan(0);
      expect(fields.length).toBeGreaterThan(0);
    });

    it('simulates job creation and completion in demo mode', async () => {
      ApiService.setDemoMode(true);
      const job = await ApiService.createJob('sc_enmap_punjab_01', 'fld_punjab_corn_01');

      expect(job.scene_id).toBe('sc_enmap_punjab_01');
      expect(job.status).toBe('queued');

      const finishedJob = await ApiService.getJob(job.job_id);
      expect(finishedJob.status).toBe('succeeded');
      expect(finishedJob.progress).toBe(1.0);
    });

    it('returns contract-compliant C4 zones in demo mode', async () => {
      ApiService.setDemoMode(true);
      const zones = await ApiService.getZones('job_demo_sample');

      expect(zones.type).toBe('FeatureCollection');
      expect(Array.isArray(zones.features)).toBe(true);
      expect(zones.features.length).toBeGreaterThan(0);

      // Verify Contract C4 fields on all features
      zones.features.forEach((feature) => {
        expect(feature.type).toBe('Feature');
        expect(feature.geometry.type).toBe('Polygon');
        const p = feature.properties;
        expect(p.zone_id).toBeDefined();
        expect(typeof p.risk_class).toBe('number');
        expect(p.risk_class).toBeGreaterThanOrEqual(0);
        expect(p.risk_class).toBeLessThanOrEqual(3);
        expect(typeof p.risk_score).toBe('number');
        expect(p.risk_score).toBeGreaterThanOrEqual(0.0);
        expect(p.risk_score).toBeLessThanOrEqual(1.0);
        expect(typeof p.area_acres).toBe('number');
        expect(typeof p.days_to_onset).toBe('number');
        expect(p.days_to_onset).toBeGreaterThanOrEqual(0.0);
        expect(p.days_to_onset).toBeLessThanOrEqual(30.0);
        expect(typeof p.dominant_indicator).toBe('string');
        expect(typeof p.recommended_action).toBe('string');
      });
    });
  });

  describe('Offline & Failure Graceful Degradation', () => {
    it('gracefully degrades health check to offline on fetch failure', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const health = await ApiService.checkHealth();
      expect(health.status).toBe('offline');
      expect(health.cuda).toBe(false);
    });

    it('falls back to mock scenes and fields when live API is unavailable', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Connection refused'));

      const scenes = await ApiService.getScenes();
      const fields = await ApiService.getFields();

      expect(scenes).toEqual(SAMPLE_SCENES);
      expect(fields).toEqual(SAMPLE_FIELDS);
    });

    it('falls back to simulated job when job creation request fails', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('500 Internal Error'));

      const job = await ApiService.createJob('sc_test', 'fld_test');
      expect(job.status).toBe('queued');
      expect(job.scene_id).toBe('sc_test');
    });
  });

  describe('URL Generators', () => {
    it('generates standard XYZ tile endpoints compliant with Contract C3', () => {
      const tileUrl = ApiService.getTileUrl('job_xyz_123');
      expect(tileUrl).toContain('/v1/jobs/job_xyz_123/tiles/{z}/{x}/{y}.png');
    });

    it('generates risk GeoTIFF download URL compliant with Contract C3', () => {
      const riskUrl = ApiService.getRiskDownloadUrl('job_xyz_123');
      expect(riskUrl).toContain('/v1/jobs/job_xyz_123/risk.tif');
    });
  });
});
