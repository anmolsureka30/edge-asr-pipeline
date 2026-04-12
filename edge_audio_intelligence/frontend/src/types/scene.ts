export interface Source {
  position: [number, number, number];
  signal_type: string;
  audio_path?: string;
  frequency: number;
  amplitude: number;
  label: string;
  transcription?: string;
  onset_s: number;
  offset_s: number;
}

export interface MicArray {
  array_type: string;
  n_mics: number;
  spacing: number;
  center: [number, number];
  height: number;
}

export interface SceneConfig {
  room_dim: [number, number, number];
  rt60: number;
  snr_db: number;
  sources: Source[];
  mic_array: MicArray;
  duration_s: number;
  fs: number;
  noise_type: string;
  seed: number;
}

export interface SceneResponse {
  scene_id: string;
  n_sources: number;
  true_doas: number[];
  transcriptions: string[];
  mic_positions: number[][];
  duration_s: number;
  sample_rate: number;
}
