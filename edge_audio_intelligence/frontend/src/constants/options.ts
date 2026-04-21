export const SSL_OPTIONS = [
  { label: 'GCC-PHAT (recommended)', value: 'gcc_phat' },
  { label: 'SRP-PHAT', value: 'srp_phat' },
];

export const BF_OPTIONS = [
  { label: 'Delay-and-Sum', value: 'delay_and_sum' },
  { label: 'MVDR', value: 'mvdr' },
  { label: 'None (skip)', value: 'none' },
];

export const ENH_OPTIONS = [
  { label: 'Spectral Subtraction', value: 'spectral_subtraction' },
  { label: 'Wavelet Enhancement', value: 'wavelet_enhancement' },
  { label: 'None (skip)', value: 'none' },
];

export const ASR_OPTIONS = [
  { label: 'Whisper tiny', value: 'whisper_tiny' },
  { label: 'Whisper base', value: 'whisper_base' },
  { label: 'Whisper small', value: 'whisper_small' },
  { label: 'None (skip)', value: 'none' },
];

export const VAD_OPTIONS = [
  { label: 'None (skip)', value: 'none' },
  { label: 'Full VAD (AtomicVAD + TEN VAD)', value: 'full_vad' },
  { label: 'TEN VAD only', value: 'ten_vad' },
  { label: 'AtomicVAD only', value: 'atomic_vad' },
  { label: 'Wavelet Energy (EE678)', value: 'wavelet_vad' },
];

export const DIARIZATION_OPTIONS = [
  { label: 'None (skip)', value: 'none' },
  { label: 'Pyannote 3.1 (recommended)', value: 'pyannote_3.1' },
  { label: 'Pyannote 3.0', value: 'pyannote_3.0' },
];

export const ROLE_OPTIONS = [
  { label: 'Target Speaker', value: 'target_speaker' },
  { label: 'Interfering Speaker', value: 'interfering_speaker' },
  { label: 'Background Music', value: 'background_music' },
  { label: 'Ambient Noise', value: 'ambient_noise' },
];

export const SIGNAL_TYPE_OPTIONS = [
  { label: 'LibriSpeech', value: 'speech' },
  { label: 'Sine wave', value: 'sine' },
  { label: 'White noise', value: 'white_noise' },
  { label: 'Chirp', value: 'chirp' },
];

export const MIC_ARRAY_OPTIONS = [
  { label: 'Edge device 4-mic linear (15mm)', value: 'linear', n_mics: 4, spacing: 0.015 },
  { label: 'Smart speaker 4-mic circular (58mm)', value: 'smart_speaker_4mic', n_mics: 4, spacing: 0.029 },
  { label: 'Phone 2-mic (14mm)', value: 'phone_2mic', n_mics: 2, spacing: 0.014 },
];

export const DISPLAY_NAMES: Record<string, string> = {
  gcc_phat: 'GCC-PHAT', srp_phat: 'SRP-PHAT',
  delay_and_sum: 'Delay-Sum', mvdr: 'MVDR',
  spectral_subtraction: 'Spectral Sub', wavelet_enhancement: 'Wavelet Enh',
  whisper_tiny: 'Whisper tiny', whisper_base: 'Whisper base', whisper_small: 'Whisper small',
  full_vad: 'AtomicVAD+TEN', ten_vad: 'TEN VAD', atomic_vad: 'AtomicVAD', wavelet_vad: 'Wavelet VAD',
  pyannote_3_1: 'Pyannote 3.1', pyannote_3_0: 'Pyannote 3.0',
  'pyannote_3.1': 'Pyannote 3.1', 'pyannote_3.0': 'Pyannote 3.0',
  none: 'skip',
};

export const STAGE_COLORS: Record<string, string> = {
  vad: '#7B1FA2',
  ssl: '#E8913A',
  bf: '#3A7EE8',
  enh: '#4CAF50',
  diarization: '#00897B',
  asr: '#E53935',
};

export const ROLE_COLORS: Record<string, string> = {
  target_speaker: '#ef4444',
  interfering_speaker: '#f97316',
  background_music: '#a855f7',
  ambient_noise: '#6b7280',
};
