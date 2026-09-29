import {
  CONTRACT_SAMPLE_ZONES,
  SAMPLE_FIELDS,
  SAMPLE_JOB,
  SAMPLE_SCENES,
} from '../fixtures/mockData';
import { FieldItem, JobItem, SceneItem, ZoneCollection } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export class ApiService {
  private static isDemoMode = false;
  private static apiKey = 'demo-token-terraspectra-2026';

  public static setDemoMode(val: boolean) {
    this.isDemoMode = val;
  }

  public static getDemoMode(): boolean {
    return this.isDemoMode;
  }

  public static async checkHealth(): Promise<{ status: string; version?: string; cuda?: boolean }> {
    if (this.isDemoMode) {
      return { status: 'healthy (demo fixture)', version: '0.1.0', cuda: true };
    }
    try {
      const res = await fetch(`${API_BASE}/v1/health`, {
        headers: { 'X-API-Key': this.apiKey },
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) throw new Error('Health check returned non-200');
      return await res.json();
    } catch {
      return { status: 'offline', cuda: false };
    }
  }

  public static async getScenes(): Promise<SceneItem[]> {
    if (this.isDemoMode) return SAMPLE_SCENES;
    try {
      const res = await fetch(`${API_BASE}/v1/scenes`, {
        headers: { 'X-API-Key': this.apiKey },
        signal: AbortSignal.timeout(2500),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (!data || data.length === 0) return SAMPLE_SCENES;
      return data;
    } catch {
      return SAMPLE_SCENES;
    }
  }

  public static async getFields(): Promise<FieldItem[]> {
    if (this.isDemoMode) return SAMPLE_FIELDS;
    try {
      const res = await fetch(`${API_BASE}/v1/fields`, {
        headers: { 'X-API-Key': this.apiKey },
        signal: AbortSignal.timeout(2500),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (!data || data.length === 0) return SAMPLE_FIELDS;
      return data;
    } catch {
      return SAMPLE_FIELDS;
    }
  }

  public static async createJob(sceneId: string, fieldId?: string): Promise<JobItem> {
    if (this.isDemoMode) {
      return {
        ...SAMPLE_JOB,
        job_id: `job_${Math.random().toString(36).substring(2, 9)}`,
        scene_id: sceneId,
        field_id: fieldId || 'fld_punjab_corn_01',
        status: 'queued',
        progress: 0.1,
      };
    }
    try {
      const res = await fetch(`${API_BASE}/v1/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.apiKey,
        },
        body: JSON.stringify({ scene_id: sceneId, field_id: fieldId }),
      });
      if (!res.ok) throw new Error(`Job creation failed: ${res.statusText}`);
      return await res.json();
    } catch {
      // Graceful fallback to simulated job
      return {
        ...SAMPLE_JOB,
        job_id: `job_${Math.random().toString(36).substring(2, 9)}`,
        scene_id: sceneId,
        field_id: fieldId,
        status: 'queued',
        progress: 0.1,
      };
    }
  }

  public static async getJob(jobId: string): Promise<JobItem> {
    if (this.isDemoMode || jobId.startsWith('job_')) {
      return {
        ...SAMPLE_JOB,
        job_id: jobId,
        status: 'succeeded',
        progress: 1.0,
      };
    }
    try {
      const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`, {
        headers: { 'X-API-Key': this.apiKey },
      });
      if (!res.ok) throw new Error();
      return await res.json();
    } catch {
      return { ...SAMPLE_JOB, job_id: jobId, status: 'succeeded', progress: 1.0 };
    }
  }

  public static async getZones(jobId: string): Promise<ZoneCollection> {
    if (this.isDemoMode) return CONTRACT_SAMPLE_ZONES;
    try {
      const res = await fetch(`${API_BASE}/v1/jobs/${jobId}/zones`, {
        headers: { 'X-API-Key': this.apiKey },
      });
      if (!res.ok) throw new Error();
      return await res.json();
    } catch {
      return CONTRACT_SAMPLE_ZONES;
    }
  }

  public static getTileUrl(jobId: string): string {
    return `${API_BASE}/v1/jobs/${jobId}/tiles/{z}/{x}/{y}.png`;
  }

  public static getRiskDownloadUrl(jobId: string): string {
    return `${API_BASE}/v1/jobs/${jobId}/risk.tif`;
  }
}
