export interface PipelineConfig {
  scene_id: string;
  ssl: string;
  bf: string;
  enh: string;
  asr: string;
  vad: string;
  enh_gate: boolean;
}

export interface StageProgress {
  stage: string;
  progress: number;
  message: string;
  latency_ms: number;
  metrics: Record<string, number>;
}

export interface StageResult {
  stage: string;
  method: string;
  latency_ms: number;
  metrics: Record<string, number>;
}

export interface PipelineResult {
  run_id: string;
  scene_id: string;
  stages: StageResult[];
  audio_signals: string[];
  transcriptions: string[];
  metrics: Record<string, number>;
  total_latency_ms: number;
  total_rtf: number;
  plots: Record<string, unknown>;
  module_names: Record<string, string>;
  vad_method: string;
  enhancement_applied: boolean;
  enhancement_gate_reason: string;
}
