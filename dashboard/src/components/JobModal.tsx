import React, { useState } from 'react';
import { 
  CheckCircle2, 
  Cpu, 
  FileText, 
  Loader2, 
  Play, 
  Satellite, 
  Sparkles, 
  X 
} from 'lucide-react';
import { FieldItem, SceneItem } from '../types';
import { ApiService } from '../services/api';

interface JobModalProps {
  isOpen: boolean;
  onClose: () => void;
  scenes: SceneItem[];
  fields: FieldItem[];
  onJobComplete: (jobId: string) => void;
}

export const JobModal: React.FC<JobModalProps> = ({
  isOpen,
  onClose,
  scenes,
  fields,
  onJobComplete,
}) => {
  const [selectedScene, setSelectedScene] = useState<string>(scenes[0]?.scene_id || '');
  const [selectedField, setSelectedField] = useState<string>(fields[0]?.id || '');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [activeStepText, setActiveStepText] = useState<string>('');
  const [isStreamActive, setIsStreamActive] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleStartScan = async () => {
    setIsProcessing(true);
    setProgressPercent(5);
    setActiveStepText('Connecting to inference queue & initializing pipeline...');
    setIsStreamActive(true);
    setErrorMessage(null);

    try {
      // 1. Create or register inference job
      const job = await ApiService.createJob(selectedScene, selectedField);

      // 2. Subscribe to real-time Server-Sent Events (SSE)
      const unsubscribe = ApiService.subscribeJobEvents(
        job.job_id,
        (evt) => {
          if (evt.step) {
            setActiveStepText(evt.step);
          }
          if (typeof evt.progress === 'number') {
            setProgressPercent(Math.min(100, Math.max(5, Math.round(evt.progress * 100))));
          }
          if (evt.status === 'succeeded') {
            setProgressPercent(100);
            setActiveStepText('Inference complete! Loading risk zones and tiles...');
            setTimeout(() => {
              setIsProcessing(false);
              setIsStreamActive(false);
              onJobComplete(job.job_id);
              onClose();
            }, 600);
          } else if (evt.status === 'failed') {
            setIsProcessing(false);
            setIsStreamActive(false);
            setErrorMessage(evt.error || 'Job failed during inference');
          }
        },
        (err) => {
          console.warn('SSE subscription error:', err);
        }
      );
    } catch (err: unknown) {
      setIsProcessing(false);
      setIsStreamActive(false);
      setErrorMessage(err instanceof Error ? err.message : 'Failed to launch scan');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in select-none">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">
                Launch Hyperspectral Pre-Visual Scan
              </h3>
              <p className="text-xs text-slate-400">
                End-to-End P1 Pipeline + P2 Model + P3 API Inference
              </p>
            </div>
          </div>
          {!isProcessing && (
            <button
              onClick={onClose}
              className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Form Body */}
        <div className="p-6 space-y-4">
          {/* Scene Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
              <Satellite className="w-3.5 h-3.5 text-emerald-400" />
              Hyperspectral Satellite Scene (Contract C1)
            </label>
            <select
              disabled={isProcessing}
              value={selectedScene}
              onChange={(e) => setSelectedScene(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 transition-colors disabled:opacity-60"
            >
              {scenes.map((s) => (
                <option key={s.scene_id} value={s.scene_id}>
                  {s.name} ({s.satellite} &bull; {s.bands} bands &bull; {s.crs})
                </option>
              ))}
            </select>
          </div>

          {/* Farm Field Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-emerald-400" />
              Target Field Boundary AOI
            </label>
            <select
              disabled={isProcessing}
              value={selectedField}
              onChange={(e) => setSelectedField(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 transition-colors disabled:opacity-60"
            >
              {fields.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} — {f.total_acres} ac ({f.crop_type})
                </option>
              ))}
            </select>
          </div>

          {/* Processing Progress Bar & Status */}
          {isProcessing && (
            <div className="pt-2 space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-emerald-400 flex items-center gap-2 font-medium truncate max-w-[360px]">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400 shrink-0" />
                  <span className="truncate">{activeStepText || 'Processing hyperspectral scan...'}</span>
                </span>
                <span className="text-emerald-400 font-mono font-bold shrink-0">
                  {progressPercent}%
                </span>
              </div>
              <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-emerald-600 via-teal-400 to-emerald-300 rounded-full transition-all duration-300 shadow-sm shadow-emerald-500/50"
                  style={{
                    width: `${progressPercent}%`,
                  }}
                />
              </div>
              {isStreamActive && (
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                  <span className="flex items-center gap-1.5 text-emerald-400/90 font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping inline-block" />
                    SSE Telemetry Connected
                  </span>
                  <span className="text-slate-500 font-mono">Channel: live-stream</span>
                </div>
              )}
            </div>
          )}

          {errorMessage && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
              {errorMessage}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-end gap-2.5">
          <button
            disabled={isProcessing}
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            disabled={isProcessing}
            onClick={handleStartScan}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30 flex items-center gap-2 transition-all disabled:opacity-50"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Running Scan...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Run Inference</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
