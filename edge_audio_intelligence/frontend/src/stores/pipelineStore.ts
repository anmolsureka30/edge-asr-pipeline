import { create } from 'zustand';
import { PIPELINE_PRESETS } from '../constants/presets';

interface PipelineState {
  preset: string;
  ssl: string;
  bf: string;
  enh: string;
  asr: string;
  vad: string;
  diarization: string;
  enhGate: boolean;

  setSsl: (v: string) => void;
  setBf: (v: string) => void;
  setEnh: (v: string) => void;
  setAsr: (v: string) => void;
  setVad: (v: string) => void;
  setDiarization: (v: string) => void;
  setEnhGate: (v: boolean) => void;
  applyPreset: (key: string) => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  preset: 'custom',
  ssl: 'gcc_phat',
  bf: 'delay_and_sum',
  enh: 'spectral_subtraction',
  asr: 'none',
  vad: 'none',
  diarization: 'none',
  enhGate: false,

  setSsl: (v) => set({ ssl: v, preset: 'custom' }),
  setBf: (v) => set({ bf: v, preset: 'custom' }),
  setEnh: (v) => set({ enh: v, preset: 'custom' }),
  setAsr: (v) => set({ asr: v, preset: 'custom' }),
  setVad: (v) => set({ vad: v, preset: 'custom' }),
  setDiarization: (v) => set({ diarization: v, preset: 'custom' }),
  setEnhGate: (v) => set({ enhGate: v, preset: 'custom' }),

  applyPreset: (key) => {
    const p = PIPELINE_PRESETS[key];
    if (p) {
      set({
        preset: key, ssl: p.ssl, bf: p.bf, enh: p.enh,
        asr: p.asr, vad: p.vad, diarization: p.diarization, enhGate: p.enh_gate,
      });
    }
  },
}));
