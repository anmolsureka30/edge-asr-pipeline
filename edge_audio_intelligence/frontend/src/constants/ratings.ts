export type RatingLevel = 'Excellent' | 'Good' | 'Acceptable' | 'Poor';

interface RatingThreshold {
  thresholds: [number, RatingLevel][];
  direction: 'lower_better' | 'higher_better';
  unit: string;
  format: (v: number) => string;
}

export const RATING_THRESHOLDS: Record<string, RatingThreshold> = {
  angular_error_deg: {
    thresholds: [[5, 'Excellent'], [10, 'Good'], [20, 'Acceptable'], [999, 'Poor']],
    direction: 'lower_better', unit: '\u00b0',
    format: (v) => `${v.toFixed(1)}\u00b0`,
  },
  pesq: {
    thresholds: [[3.5, 'Excellent'], [2.5, 'Good'], [2.0, 'Acceptable'], [-1, 'Poor']],
    direction: 'higher_better', unit: '',
    format: (v) => v.toFixed(2),
  },
  stoi: {
    thresholds: [[0.9, 'Excellent'], [0.8, 'Good'], [0.7, 'Acceptable'], [-1, 'Poor']],
    direction: 'higher_better', unit: '',
    format: (v) => v.toFixed(3),
  },
  si_sdr_db: {
    thresholds: [[15, 'Excellent'], [10, 'Good'], [5, 'Acceptable'], [-99, 'Poor']],
    direction: 'higher_better', unit: 'dB',
    format: (v) => `${v.toFixed(1)} dB`,
  },
  si_sdr_improvement_db: {
    thresholds: [[10, 'Excellent'], [5, 'Good'], [0, 'Acceptable'], [-99, 'Poor']],
    direction: 'higher_better', unit: 'dB',
    format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)} dB`,
  },
  wer_avg: {
    thresholds: [[0.05, 'Excellent'], [0.15, 'Good'], [0.30, 'Acceptable'], [99, 'Poor']],
    direction: 'lower_better', unit: '%',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  cer: {
    thresholds: [[0.03, 'Excellent'], [0.10, 'Good'], [0.20, 'Acceptable'], [99, 'Poor']],
    direction: 'lower_better', unit: '%',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  rtf: {
    thresholds: [[0.5, 'Excellent'], [1.0, 'Good'], [2.0, 'Acceptable'], [99, 'Poor']],
    direction: 'lower_better', unit: '',
    format: (v) => v.toFixed(3),
  },
};

export function getRating(metricKey: string, value: number | null | undefined): RatingLevel | null {
  if (value == null || isNaN(value)) return null;
  const config = RATING_THRESHOLDS[metricKey];
  if (!config) return null;

  if (config.direction === 'lower_better') {
    for (const [threshold, rating] of config.thresholds) {
      if (value <= threshold) return rating;
    }
    return 'Poor';
  } else {
    // higher_better: thresholds are in descending order
    for (const [threshold, rating] of config.thresholds) {
      if (value >= threshold) return rating;
    }
    return 'Poor';
  }
}

export function formatMetric(metricKey: string, value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '\u2014';
  const config = RATING_THRESHOLDS[metricKey];
  if (config) return config.format(value);
  if (metricKey.includes('latency') || metricKey.includes('ms')) return `${value.toFixed(1)} ms`;
  return value.toFixed(3);
}

export const RATING_COLORS: Record<RatingLevel, { bg: string; text: string; border: string }> = {
  Excellent: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  Good: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  Acceptable: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  Poor: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
};

// Stage grouping for hierarchical metrics table
export const METRIC_STAGE_MAP: Record<string, string> = {
  angular_error_deg: 'SSL',
  si_sdr_db: 'Beamforming',
  si_sdr_improvement_db: 'Beamforming',
  pesq: 'Enhancement',
  stoi: 'Enhancement',
  wer_avg: 'ASR',
  wer_source_0: 'ASR',
  wer_source_1: 'ASR',
  cer: 'ASR',
};

export const METRIC_DISPLAY_NAMES: Record<string, string> = {
  angular_error_deg: 'Angular Error',
  si_sdr_db: 'SI-SDR',
  si_sdr_improvement_db: 'SI-SDR Improvement',
  pesq: 'PESQ',
  stoi: 'STOI',
  wer_avg: 'WER (avg)',
  wer_source_0: 'WER (source 0)',
  wer_source_1: 'WER (source 1)',
  cer: 'CER',
  rtf: 'Real-Time Factor',
  speech_pct: 'Speech %',
  speech_frames: 'Speech Frames',
  noise_frames: 'Noise Frames',
  n_segments: 'Segments',
};
