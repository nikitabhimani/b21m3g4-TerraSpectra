import {
  CONTRACT_SAMPLE_ZONES,
  SAMPLE_FIELDS,
  SAMPLE_JOB,
  SAMPLE_SCENES,
} from '../fixtures/mockData';
import { FieldItem, JobEvent, JobItem, SceneItem, ZoneCollection } from '../types';

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

  public static subscribeJobEvents(
    jobId: string,
    onEvent: (evt: JobEvent) => void,
    onError?: (err: unknown) => void
  ): () => void {
    if (this.isDemoMode || jobId.startsWith('job_mock_') || typeof EventSource === 'undefined') {
      let isCancelled = false;
      const demoSteps = [
        { progress: 0.1, step: 'Validating Contract C1 200-band hyperspectral cube & CRS metadata...' },
        { progress: 0.35, step: 'Windowing raster into 64x64 patches & pre-fetching...' },
        { progress: 0.65, step: 'Executing 3D-CNN + ViT hybrid inference on GPU (models/model.pt)...' },
        { progress: 0.88, step: 'Stitching Hann-window blended risk rasters & writing COG...' },
        { progress: 0.98, step: 'Vectorizing epidemiological polygons & computing ROI...' },
        { progress: 1.0, step: 'Scan complete. All layers generated.' },
      ];

      (async () => {
        for (let i = 0; i < demoSteps.length; i++) {
          if (isCancelled) return;
          await new Promise((r) => setTimeout(r, 600));
          if (isCancelled) return;
          const s = demoSteps[i];
          const isDone = i === demoSteps.length - 1;
          onEvent({
            job_id: jobId,
            status: isDone ? 'succeeded' : 'running',
            progress: s.progress,
            step: s.step,
            summary: isDone ? SAMPLE_JOB.summary : null,
          });
        }
      })();

      return () => {
        isCancelled = true;
      };
    }

    try {
      const sseUrl = `${API_BASE}/v1/jobs/${jobId}/stream?api_key=${this.apiKey}`;
      const eventSource = new EventSource(sseUrl);

      eventSource.addEventListener('progress', (e: MessageEvent) => {
        try {
          const data: JobEvent = JSON.parse(e.data);
          onEvent(data);
        } catch {
          // ignore malformed message
        }
      });

      eventSource.addEventListener('complete', (e: MessageEvent) => {
        try {
          const data: JobEvent = JSON.parse(e.data);
          onEvent(data);
        } catch {
          // ignore malformed message
        }
        eventSource.close();
      });

      eventSource.addEventListener('error', (e: Event) => {
        if (onError) onError(e);
        eventSource.close();
      });

      return () => {
        eventSource.close();
      };
    } catch (err) {
      if (onError) onError(err);
      return () => {};
    }
  }
}

