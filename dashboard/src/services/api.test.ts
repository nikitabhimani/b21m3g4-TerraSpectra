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

  describe('Real-Time SSE Job Events Streaming', () => {
    it('streams job progress events in demo mode', async () => {
      ApiService.setDemoMode(true);
      const events: any[] = [];
      const unsubscribe = ApiService.subscribeJobEvents('job_test_stream', (evt) => {
        events.push(evt);
      });

      // Wait for at least 2 events to fire
      await new Promise((r) => setTimeout(r, 1300));
      unsubscribe();

      expect(events.length).toBeGreaterThanOrEqual(1);
      expect(events[0].job_id).toBe('job_test_stream');
      expect(events[0].progress).toBeGreaterThan(0);
      expect(events[0].step).toBeDefined();
    });

    it('supports immediate cancellation via unsubscribe callback', async () => {
      ApiService.setDemoMode(true);
      const events: any[] = [];
      const unsubscribe = ApiService.subscribeJobEvents('job_cancel_stream', (evt) => {
        events.push(evt);
      });
      // Cancel immediately
      unsubscribe();

      await new Promise((r) => setTimeout(r, 800));
      expect(events.length).toBe(0);
    });
  });

  describe('Job Lifecycle Control: Cancellation & Retry', () => {
    it('cancels a job in demo mode', async () => {
      ApiService.setDemoMode(true);
      const res = await ApiService.cancelJob('job_cancel_demo_1');
      expect(res.job_id).toBe('job_cancel_demo_1');
      expect(res.status).toBe('cancelled');
      expect(res.error).toBe('Cancelled by user');
    });

    it('cancels a job via API request when online', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          job_id: 'job_live_01',
          scene_id: 'scn_1',
          status: 'cancelled',
          progress: 0.45,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          error: 'Cancelled by user',
        }),
      });

      const res = await ApiService.cancelJob('job_live_01');
      expect(res.status).toBe('cancelled');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/jobs/job_live_01/cancel'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('retries a cancelled or failed job in demo mode', async () => {
      ApiService.setDemoMode(true);
      const res = await ApiService.retryJob('job_retry_demo_1');
      expect(res.job_id).toBe('job_retry_demo_1');
      expect(res.status).toBe('queued');
      expect(res.progress).toBe(0.05);
      expect(res.error).toBeNull();
    });

    it('retries a job via API request when online', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          job_id: 'job_live_02',
          scene_id: 'scn_1',
          status: 'queued',
          progress: 0.0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });

      const res = await ApiService.retryJob('job_live_02');
      expect(res.status).toBe('queued');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/jobs/job_live_02/retry'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });
});
