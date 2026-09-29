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
  const [progressStep, setProgressStep] = useState(0);

  if (!isOpen) return null;

  const pipelineSteps = [
    'Validating Contract C1 200-Band Cube & CRS Metadata...',
    'Windowing into 64x64 Patches & Radiometric Robust Scaling...',
    'Running 3D-CNN + ViT Hybrid Model Inference (models/model.pt)...',
    'Extracting Dominant Indicators via Integrated Gradients (Contract C4)...',
    'Executing Hann-Window Blended Stitching & Vector Polygonization...',
  ];

  const handleStartScan = async () => {
    setIsProcessing(true);
    setProgressStep(0);

    // Call API service to register/create job
    const job = await ApiService.createJob(selectedScene, selectedField);

    // Step-by-step progress simulation representing live pipeline stages
    for (let i = 0; i < pipelineSteps.length; i++) {
      setProgressStep(i);
      await new Promise((r) => setTimeout(r, 650));
    }

    setIsProcessing(false);
    onJobComplete(job.job_id);
    onClose();
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
                <span className="text-emerald-400 flex items-center gap-2 font-medium">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400" />
                  {pipelineSteps[progressStep]}
                </span>
                <span className="text-slate-400 font-mono">
                  {Math.round(((progressStep + 1) / pipelineSteps.length) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-emerald-600 to-teal-400 rounded-full transition-all duration-300"
                  style={{
                    width: `${((progressStep + 1) / pipelineSteps.length) * 100}%`,
                  }}
                />
              </div>
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
