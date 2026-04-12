import { create } from 'zustand';
import type { StageProgress, PipelineResult } from '../types/pipeline';

interface RunState {
  status: 'idle' | 'running' | 'complete' | 'error';
  taskId: string | null;
  progress: number;
  progressMessage: string;
  stages: StageProgress[];
  result: PipelineResult | null;
  error: string | null;

  setRunning: (taskId: string) => void;
  addProgress: (stage: StageProgress) => void;
  setComplete: (result: PipelineResult) => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useRunStore = create<RunState>((set) => ({
  status: 'idle',
  taskId: null,
  progress: 0,
  progressMessage: '',
  stages: [],
  result: null,
  error: null,

  setRunning: (taskId) => set({
    status: 'running', taskId, progress: 0, progressMessage: 'Starting...',
    stages: [], result: null, error: null,
  }),

  addProgress: (stage) => set((s) => ({
    stages: [...s.stages, stage],
    progress: stage.progress,
    progressMessage: stage.message || stage.stage,
  })),

  setComplete: (result) => set({
    status: 'complete', progress: 1, progressMessage: 'Complete', result,
  }),

  setError: (error) => set({
    status: 'error', progressMessage: error, error,
  }),

  reset: () => set({
    status: 'idle', taskId: null, progress: 0, progressMessage: '',
    stages: [], result: null, error: null,
  }),
}));
