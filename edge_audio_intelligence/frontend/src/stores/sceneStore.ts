import { create } from 'zustand';
import type { Source, MicArray } from '../types/scene';
import { SCENE_PRESETS } from '../constants/presets';

export interface SourceWithRole extends Source {
  role: string;
}

interface SceneState {
  preset: string;
  roomWidth: number;
  roomDepth: number;
  roomHeight: number;
  rt60: number;
  snrDb: number;
  durationS: number;
  noiseType: string;
  sources: SourceWithRole[];
  selectedSourceIndex: number;
  micArray: MicArray;
  sceneId: string | null;
  sceneGenerated: boolean;
  trueDoas: number[];
  transcriptions: string[];

  setRoom: (p: Partial<{ roomWidth: number; roomDepth: number; roomHeight: number }>) => void;
  setRt60: (v: number) => void;
  setSnrDb: (v: number) => void;
  setDurationS: (v: number) => void;
  addSource: () => void;
  removeSource: (i: number) => void;
  updateSource: (i: number, p: Partial<SourceWithRole>) => void;
  setSelectedSource: (i: number) => void;
  setMicArray: (m: Partial<MicArray>) => void;
  setSceneId: (id: string | null) => void;
  setSceneResult: (id: string, doas: number[], trans: string[]) => void;
  loadPreset: (key: string) => void;
}

const DEFAULT_SOURCE: SourceWithRole = {
  position: [2, 3.5, 1.5], signal_type: 'speech', frequency: 440,
  amplitude: 1.0, label: 'S0', onset_s: 0, offset_s: -1, role: 'target_speaker',
};

export const useSceneStore = create<SceneState>((set) => ({
  preset: 'clean_room',
  roomWidth: 4, roomDepth: 3.5, roomHeight: 2.8,
  rt60: 0.0, snrDb: 30, durationS: 8, noiseType: 'white',
  sources: [{ ...DEFAULT_SOURCE }],
  selectedSourceIndex: 0,
  micArray: { array_type: 'linear', n_mics: 4, spacing: 0.015, center: [2, 1.75], height: 1.2 },
  sceneId: null, sceneGenerated: false, trueDoas: [], transcriptions: [],

  setRoom: (p) => set((s) => ({ ...s, ...p, sceneGenerated: false })),
  setRt60: (v) => set({ rt60: v, sceneGenerated: false }),
  setSnrDb: (v) => set({ snrDb: v, sceneGenerated: false }),
  setDurationS: (v) => set({ durationS: v, sceneGenerated: false }),

  addSource: () => set((s) => {
    const i = s.sources.length;
    return {
      sources: [...s.sources, {
        ...DEFAULT_SOURCE, label: `S${i}`,
        position: [s.roomWidth * 0.6, s.roomDepth * 0.3, 1.5],
        role: i === 0 ? 'target_speaker' : 'interfering_speaker',
      }],
      sceneGenerated: false,
    };
  }),

  removeSource: (i) => set((s) => ({
    sources: s.sources.filter((_, idx) => idx !== i),
    selectedSourceIndex: Math.max(0, Math.min(s.selectedSourceIndex, s.sources.length - 2)),
    sceneGenerated: false,
  })),

  updateSource: (i, p) => set((s) => ({
    sources: s.sources.map((src, idx) => idx === i ? { ...src, ...p } : src),
    sceneGenerated: false,
  })),

  setSelectedSource: (i) => set({ selectedSourceIndex: i }),

  setMicArray: (m) => set((s) => ({
    micArray: { ...s.micArray, ...m },
    sceneGenerated: false,
  })),

  setSceneId: (id) => set({ sceneId: id }),

  setSceneResult: (id, doas, trans) => set({
    sceneId: id, sceneGenerated: true, trueDoas: doas, transcriptions: trans,
  }),

  loadPreset: (key) => {
    const p = SCENE_PRESETS[key];
    if (!p) return;
    set({
      preset: key,
      roomWidth: p.room_dim[0], roomDepth: p.room_dim[1], roomHeight: p.room_dim[2],
      rt60: p.rt60, snrDb: p.snr_db, noiseType: p.noise_type, durationS: p.duration_s,
      sources: p.sources.map((s, i) => ({
        ...s, label: `S${i}`,
      })),
      micArray: { array_type: 'linear', n_mics: 4, spacing: 0.015,
        center: [p.room_dim[0] / 2, p.room_dim[1] / 2], height: 1.2 },
      selectedSourceIndex: 0,
      sceneId: null, sceneGenerated: false, trueDoas: [], transcriptions: [],
    });
  },
}));
