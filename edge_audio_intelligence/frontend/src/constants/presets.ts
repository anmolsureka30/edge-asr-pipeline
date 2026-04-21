import type { Source } from '../types/scene';

export interface ScenePreset {
  label: string;
  description: string;
  room_dim: [number, number, number];
  rt60: number;
  snr_db: number;
  noise_type: string;
  duration_s: number;
  sources: (Source & { role: string })[];
}

export const SCENE_PRESETS: Record<string, ScenePreset> = {
  clean_room: {
    label: 'Clean Room (anechoic, 1 speaker)',
    description: 'No reverberation, low noise. Sanity check — SSL and ASR should be near-perfect.',
    room_dim: [4.0, 3.5, 2.8],
    rt60: 0.0, snr_db: 30.0, noise_type: 'white', duration_s: 8.0,
    sources: [{
      role: 'target_speaker', signal_type: 'speech', position: [2.0, 2.5, 1.5],
      frequency: 440, amplitude: 1.0, label: 'S0', onset_s: 0.5, offset_s: 7.0,
      audio_path: undefined, transcription: undefined,
    }],
  },
  office_meeting: {
    label: 'Office Meeting (2 overlapping speakers)',
    description: 'Moderate reverb, 2 speakers with 3s overlap. Tests SSL, beamforming, VAD, and enhancement gate.',
    room_dim: [7.0, 6.0, 3.0],
    rt60: 0.4, snr_db: 20.0, noise_type: 'white', duration_s: 12.0,
    sources: [
      { role: 'target_speaker', signal_type: 'speech', position: [2.5, 4.0, 1.5],
        frequency: 440, amplitude: 1.0, label: 'S0', onset_s: 1.0, offset_s: 10.0,
        audio_path: undefined, transcription: undefined },
      { role: 'interfering_speaker', signal_type: 'speech', position: [5.0, 2.0, 1.5],
        frequency: 440, amplitude: 0.8, label: 'S1', onset_s: 5.0, offset_s: 11.0,
        audio_path: undefined, transcription: undefined },
    ],
  },
  sequential_speakers: {
    label: 'Sequential Speakers (no overlap)',
    description: 'Two speakers taking turns. Tests VAD onset/offset precision and diarization.',
    room_dim: [6.0, 5.0, 3.0],
    rt60: 0.3, snr_db: 20.0, noise_type: 'white', duration_s: 10.0,
    sources: [
      { role: 'target_speaker', signal_type: 'speech', position: [2.0, 3.5, 1.5],
        frequency: 440, amplitude: 1.0, label: 'S0', onset_s: 0.5, offset_s: 4.5,
        audio_path: undefined, transcription: undefined },
      { role: 'interfering_speaker', signal_type: 'speech', position: [4.5, 1.5, 1.5],
        frequency: 440, amplitude: 0.9, label: 'S1', onset_s: 5.5, offset_s: 9.5,
        audio_path: undefined, transcription: undefined },
    ],
  },
  noisy_cafe: {
    label: 'Noisy Cafe (hard)',
    description: 'High noise, moderate reverb, 2 speakers. Stress test for the full pipeline.',
    room_dim: [8.0, 6.0, 3.5],
    rt60: 0.5, snr_db: 5.0, noise_type: 'white', duration_s: 10.0,
    sources: [
      { role: 'target_speaker', signal_type: 'speech', position: [3.0, 4.0, 1.5],
        frequency: 440, amplitude: 1.0, label: 'S0', onset_s: 1.0, offset_s: 9.0,
        audio_path: undefined, transcription: undefined },
      { role: 'interfering_speaker', signal_type: 'speech', position: [6.0, 2.0, 1.5],
        frequency: 440, amplitude: 0.7, label: 'S1', onset_s: 3.0, offset_s: 8.0,
        audio_path: undefined, transcription: undefined },
    ],
  },
  reverberant_hall: {
    label: 'Reverberant Hall (single speaker)',
    description: 'Very high RT60 tests SSL robustness in heavy reverberation.',
    room_dim: [12.0, 8.0, 4.0],
    rt60: 1.2, snr_db: 25.0, noise_type: 'white', duration_s: 8.0,
    sources: [{
      role: 'target_speaker', signal_type: 'speech', position: [4.0, 5.0, 1.5],
      frequency: 440, amplitude: 1.0, label: 'S0', onset_s: 0.5, offset_s: 7.0,
      audio_path: undefined, transcription: undefined,
    }],
  },
};

export const PIPELINE_PRESETS: Record<string, {
  label: string;
  description: string;
  ssl: string; bf: string; enh: string; asr: string; vad: string; diarization: string; enh_gate: boolean;
}> = {
  custom: {
    label: 'Custom', description: 'Select each module manually',
    ssl: 'gcc_phat', bf: 'delay_and_sum', enh: 'spectral_subtraction',
    asr: 'none', vad: 'none', diarization: 'none', enh_gate: false,
  },
  cascade_basic: {
    label: 'Basic Cascade (no VAD)',
    description: 'GCC-PHAT \u2192 Delay-Sum \u2192 Spectral Sub \u2192 Whisper tiny. No VAD, no gating.',
    ssl: 'gcc_phat', bf: 'delay_and_sum', enh: 'spectral_subtraction',
    asr: 'whisper_tiny', vad: 'none', diarization: 'none', enh_gate: false,
  },
  vad_gated_mvdr: {
    label: 'VAD-Gated MVDR',
    description: 'TEN VAD noise labels \u2192 MVDR \u03a6_nn gating \u2192 Wavelet enhancement (gated)',
    ssl: 'gcc_phat', bf: 'mvdr', enh: 'wavelet_enhancement',
    asr: 'whisper_base', vad: 'ten_vad', diarization: 'none', enh_gate: true,
  },
  full_pipeline: {
    label: 'Full Pipeline (all stages)',
    description: 'AtomicVAD + TEN VAD \u2192 MVDR \u2192 Wavelet Enh (gated) \u2192 Pyannote 3.1 \u2192 Whisper small',
    ssl: 'gcc_phat', bf: 'mvdr', enh: 'wavelet_enhancement',
    asr: 'whisper_small', vad: 'full_vad', diarization: 'pyannote_3.1', enh_gate: true,
  },
  wavelet_baseline: {
    label: 'Wavelet Baseline (EE678)',
    description: 'Wavelet VAD \u2192 GCC-PHAT \u2192 Delay-Sum \u2192 Wavelet Enh \u2192 Whisper tiny',
    ssl: 'gcc_phat', bf: 'delay_and_sum', enh: 'wavelet_enhancement',
    asr: 'whisper_tiny', vad: 'wavelet_vad', diarization: 'none', enh_gate: false,
  },
};
