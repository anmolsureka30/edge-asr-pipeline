import { useState } from 'react';
import { useSceneStore } from '../../stores/sceneStore';
import { useQuery } from '@tanstack/react-query';
import { generateScene, getLibriSpeechIndex } from '../../api/client';
import { SCENE_PRESETS } from '../../constants/presets';
import { ROLE_OPTIONS, SIGNAL_TYPE_OPTIONS, MIC_ARRAY_OPTIONS, ROLE_COLORS } from '../../constants/options';

function Slider({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="flex justify-between text-xs text-slate-500 mb-0.5">
        <span>{label}</span>
        <span className="font-mono font-medium text-slate-700">{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
    </label>
  );
}

function SourceCard({ index }: { index: number }) {
  const { sources, updateSource, removeSource, selectedSourceIndex, setSelectedSource } = useSceneStore();
  const src = sources[index];
  const isSelected = selectedSourceIndex === index;
  const roleColor = ROLE_COLORS[src.role] || '#6b7280';

  const { data: libriIndex } = useQuery({
    queryKey: ['librispeech'],
    queryFn: () => getLibriSpeechIndex(200),
    staleTime: Infinity,
  });

  return (
    <div
      className={`border rounded-lg p-3 transition-all cursor-pointer ${
        isSelected ? 'ring-2 ring-blue-500 border-blue-300 bg-blue-50/30' : 'border-slate-200 hover:border-slate-300'
      }`}
      onClick={() => setSelectedSource(index)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: roleColor }} />
          <span className="font-medium text-sm text-slate-800">{src.label}</span>
        </div>
        {sources.length > 1 && (
          <button onClick={(e) => { e.stopPropagation(); removeSource(index); }}
            className="text-slate-400 hover:text-red-500 text-xs px-1">
            Remove
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <select value={src.role} onChange={e => updateSource(index, { role: e.target.value })}
          className="border rounded px-1.5 py-1 bg-white text-xs">
          {ROLE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>

        <select value={src.signal_type} onChange={e => updateSource(index, { signal_type: e.target.value })}
          className="border rounded px-1.5 py-1 bg-white text-xs">
          {SIGNAL_TYPE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>

      {src.signal_type === 'speech' && libriIndex && (
        <select
          className="w-full border rounded px-1.5 py-1 bg-white text-xs mt-2 truncate"
          onChange={e => {
            const utt = libriIndex.find(u => u.id === e.target.value);
            if (utt) updateSource(index, { audio_path: utt.path, transcription: utt.text });
          }}
        >
          <option value="">Select utterance...</option>
          {libriIndex.map(u => (
            <option key={u.id} value={u.id}>[{u.speaker}] {u.text.slice(0, 60)}</option>
          ))}
        </select>
      )}

      <div className="grid grid-cols-4 gap-1 mt-2">
        {['x', 'y', 'z'].map((axis, ai) => (
          <label key={axis} className="block">
            <span className="text-[10px] text-slate-400 uppercase">{axis}</span>
            <input type="number" step={0.5} value={src.position[ai]}
              onChange={e => {
                const pos = [...src.position] as [number, number, number];
                pos[ai] = +e.target.value;
                updateSource(index, { position: pos });
              }}
              className="w-full border rounded px-1 py-0.5 text-xs" />
          </label>
        ))}
        <label className="block">
          <span className="text-[10px] text-slate-400">Vol</span>
          <input type="number" step={0.1} min={0.1} max={2} value={src.amplitude}
            onChange={e => updateSource(index, { amplitude: +e.target.value })}
            className="w-full border rounded px-1 py-0.5 text-xs" />
        </label>
      </div>

      <div className="grid grid-cols-2 gap-1 mt-1">
        <label className="block">
          <span className="text-[10px] text-slate-400">Onset (s)</span>
          <input type="number" step={0.5} min={0} value={src.onset_s}
            onChange={e => updateSource(index, { onset_s: +e.target.value })}
            className="w-full border rounded px-1 py-0.5 text-xs" />
        </label>
        <label className="block">
          <span className="text-[10px] text-slate-400">Offset (s)</span>
          <input type="number" step={0.5} min={-1} value={src.offset_s}
            onChange={e => updateSource(index, { offset_s: +e.target.value })}
            className="w-full border rounded px-1 py-0.5 text-xs" />
        </label>
      </div>
    </div>
  );
}

function RoomView() {
  const { roomWidth, roomDepth, sources, micArray, selectedSourceIndex, trueDoas, updateSource, setSelectedSource } = useSceneStore();
  const svgW = 400, svgH = 300;
  const pad = 30;
  const scaleX = (x: number) => pad + (x / roomWidth) * (svgW - 2 * pad);
  const scaleY = (y: number) => svgH - pad - (y / roomDepth) * (svgH - 2 * pad);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const svgPt = pt.matrixTransform(ctm.inverse());
    const x = ((svgPt.x - pad) / (svgW - 2 * pad)) * roomWidth;
    const y = ((svgH - pad - svgPt.y) / (svgH - 2 * pad)) * roomDepth;
    if (x >= 0 && x <= roomWidth && y >= 0 && y <= roomDepth) {
      const pos = [...sources[selectedSourceIndex].position] as [number, number, number];
      pos[0] = Math.round(x * 10) / 10;
      pos[1] = Math.round(y * 10) / 10;
      updateSource(selectedSourceIndex, { position: pos });
    }
  };

  const micCx = scaleX(micArray.center[0]);
  const micCy = scaleY(micArray.center[1]);

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full bg-white rounded-lg border border-slate-200"
      onClick={handleClick} style={{ cursor: 'crosshair' }}>
      {/* Grid */}
      {Array.from({ length: Math.ceil(roomWidth) + 1 }, (_, i) => (
        <line key={`gx${i}`} x1={scaleX(i)} y1={scaleY(0)} x2={scaleX(i)} y2={scaleY(roomDepth)}
          stroke="#e5e7eb" strokeWidth={0.5} strokeDasharray="2,2" />
      ))}
      {Array.from({ length: Math.ceil(roomDepth) + 1 }, (_, i) => (
        <line key={`gy${i}`} x1={scaleX(0)} y1={scaleY(i)} x2={scaleX(roomWidth)} y2={scaleY(i)}
          stroke="#e5e7eb" strokeWidth={0.5} strokeDasharray="2,2" />
      ))}

      {/* Room boundary */}
      <rect x={scaleX(0)} y={scaleY(roomDepth)} width={scaleX(roomWidth) - scaleX(0)}
        height={scaleY(0) - scaleY(roomDepth)} fill="none" stroke="#374151" strokeWidth={2} />

      {/* DOA arrow */}
      {trueDoas.map((doa, i) => {
        const rad = (doa * Math.PI) / 180;
        const len = Math.min(roomWidth, roomDepth) * 0.3;
        return (
          <line key={`doa${i}`}
            x1={micCx} y1={micCy}
            x2={scaleX(micArray.center[0] + len * Math.cos(rad))}
            y2={scaleY(micArray.center[1] + len * Math.sin(rad))}
            stroke="#3b82f6" strokeWidth={2} strokeDasharray="4,2"
            markerEnd="url(#arrowhead)" />
        );
      })}
      <defs>
        <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
          <polygon points="0 0, 6 2, 0 4" fill="#3b82f6" />
        </marker>
      </defs>

      {/* Mic array */}
      <circle cx={micCx} cy={micCy} r={6} fill="#3b82f6" stroke="white" strokeWidth={1.5} />
      <text x={micCx} y={micCy - 10} textAnchor="middle" className="text-[8px] fill-blue-600 font-medium">
        Mics ({micArray.n_mics})
      </text>

      {/* Sources */}
      {sources.map((src, i) => {
        const cx = scaleX(src.position[0]);
        const cy = scaleY(src.position[1]);
        const color = ROLE_COLORS[src.role] || '#6b7280';
        const isSelected = i === selectedSourceIndex;
        return (
          <g key={i} onClick={(e) => { e.stopPropagation(); setSelectedSource(i); }} style={{ cursor: 'pointer' }}>
            {src.role === 'target_speaker' ? (
              <polygon
                points={`${cx},${cy - 9} ${cx + 8},${cy + 6} ${cx - 8},${cy + 6}`}
                fill={color} stroke={isSelected ? '#000' : 'white'} strokeWidth={isSelected ? 2 : 1} />
            ) : (
              <rect x={cx - 6} y={cy - 6} width={12} height={12} rx={2}
                fill={color} stroke={isSelected ? '#000' : 'white'} strokeWidth={isSelected ? 2 : 1}
                transform={`rotate(45 ${cx} ${cy})`} />
            )}
            <text x={cx} y={cy - 13} textAnchor="middle"
              className={`text-[9px] font-medium ${isSelected ? 'fill-slate-900' : 'fill-slate-600'}`}>
              {src.label}
            </text>
          </g>
        );
      })}

      {/* Axis labels */}
      <text x={svgW / 2} y={svgH - 5} textAnchor="middle" className="text-[9px] fill-slate-400">
        Width ({roomWidth}m)
      </text>
      <text x={10} y={svgH / 2} textAnchor="middle" className="text-[9px] fill-slate-400"
        transform={`rotate(-90 10 ${svgH / 2})`}>
        Depth ({roomDepth}m)
      </text>
    </svg>
  );
}

function TimelineEditor() {
  const { sources, durationS } = useSceneStore();
  const barH = 24, gap = 4, pad = 40;
  const svgH = sources.length * (barH + gap) + pad;
  const svgW = 400;
  const timeScale = (t: number) => pad + (t / durationS) * (svgW - pad - 10);

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full bg-white rounded border border-slate-200 mt-2">
      {/* Time axis */}
      {Array.from({ length: Math.ceil(durationS) + 1 }, (_, i) => (
        <g key={i}>
          <line x1={timeScale(i)} y1={0} x2={timeScale(i)} y2={svgH - 15}
            stroke="#e5e7eb" strokeWidth={0.5} />
          <text x={timeScale(i)} y={svgH - 3} textAnchor="middle" className="text-[8px] fill-slate-400">
            {i}s
          </text>
        </g>
      ))}

      {/* Source bars */}
      {sources.map((src, i) => {
        const y = i * (barH + gap) + 4;
        const onset = src.onset_s;
        const offset = src.offset_s < 0 ? durationS : src.offset_s;
        const x1 = timeScale(onset);
        const x2 = timeScale(offset);
        const color = ROLE_COLORS[src.role] || '#6b7280';

        return (
          <g key={i}>
            <text x={4} y={y + barH / 2 + 3} className="text-[9px] fill-slate-600 font-medium">
              {src.label}
            </text>
            <rect x={x1} y={y} width={Math.max(x2 - x1, 4)} height={barH}
              rx={3} fill={color} opacity={0.7} stroke={color} strokeWidth={1} />
            <text x={(x1 + x2) / 2} y={y + barH / 2 + 3} textAnchor="middle"
              className="text-[8px] fill-white font-medium">
              {onset.toFixed(1)}s — {offset.toFixed(1)}s
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function SceneSection() {
  const store = useSceneStore();
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const resp = await generateScene({
        room_dim: [store.roomWidth, store.roomDepth, store.roomHeight],
        rt60: store.rt60, snr_db: store.snrDb,
        sources: store.sources.map(s => ({
          position: s.position, signal_type: s.signal_type,
          audio_path: s.audio_path, frequency: s.frequency,
          amplitude: s.amplitude, label: s.label,
          transcription: s.transcription, onset_s: s.onset_s, offset_s: s.offset_s,
        })),
        mic_array: store.micArray,
        duration_s: store.durationS, fs: 16000, noise_type: store.noiseType, seed: 42,
      });
      store.setSceneResult(resp.scene_id, resp.true_doas, resp.transcriptions);
    } catch (err) {
      console.error('Scene generation failed:', err);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-800">Scene Setup</h2>
        <select value={store.preset} onChange={e => store.loadPreset(e.target.value)}
          className="border rounded px-2 py-1 text-sm bg-white">
          {Object.entries(SCENE_PRESETS).map(([k, v]) => (
            <option key={k} value={k}>{v.label}</option>
          ))}
        </select>
      </div>

      <div className="p-4 grid grid-cols-12 gap-4">
        {/* Left: Controls */}
        <div className="col-span-4 space-y-3">
          <Slider label="RT60" value={store.rt60} min={0} max={2} step={0.05} unit="s" onChange={store.setRt60} />
          <Slider label="SNR" value={store.snrDb} min={-5} max={40} step={1} unit=" dB" onChange={store.setSnrDb} />
          <Slider label="Duration" value={store.durationS} min={1} max={30} step={1} unit="s" onChange={store.setDurationS} />

          <div className="grid grid-cols-3 gap-1">
            {[
              { label: 'W', value: store.roomWidth, key: 'roomWidth' as const },
              { label: 'D', value: store.roomDepth, key: 'roomDepth' as const },
              { label: 'H', value: store.roomHeight, key: 'roomHeight' as const },
            ].map(({ label, value, key }) => (
              <label key={key} className="block">
                <span className="text-[10px] text-slate-400">{label} (m)</span>
                <input type="number" step={0.5} min={2} max={20} value={value}
                  onChange={e => store.setRoom({ [key]: +e.target.value })}
                  className="w-full border rounded px-1.5 py-1 text-xs" />
              </label>
            ))}
          </div>

          {/* Mic array */}
          <select
            value={MIC_ARRAY_OPTIONS.find(o => o.value === store.micArray.array_type)?.value || 'linear'}
            onChange={e => {
              const opt = MIC_ARRAY_OPTIONS.find(o => o.value === e.target.value);
              if (opt) store.setMicArray({
                array_type: opt.value, n_mics: opt.n_mics, spacing: opt.spacing,
                center: [store.roomWidth / 2, store.roomDepth / 2],
              });
            }}
            className="w-full border rounded px-2 py-1.5 text-xs bg-white"
          >
            {MIC_ARRAY_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>

          <button onClick={handleGenerate} disabled={generating}
            className={`w-full rounded-lg px-4 py-2.5 text-sm font-semibold text-white transition-all
              ${generating ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700 shadow-sm hover:shadow'}`}>
            {generating ? 'Generating...' : store.sceneGenerated ? 'Regenerate Scene' : 'Generate Scene'}
          </button>

          {store.sceneGenerated && (
            <div className="text-xs text-emerald-600 bg-emerald-50 rounded px-2 py-1.5 border border-emerald-200">
              Scene ready ({store.sceneId}) — {store.trueDoas.length} source(s),
              DOA: {store.trueDoas.map(d => `${d.toFixed(1)}\u00b0`).join(', ')}
            </div>
          )}
        </div>

        {/* Center: Room view */}
        <div className="col-span-5">
          <RoomView />
          <p className="text-[10px] text-slate-400 mt-1 text-center">
            Click to place selected source. Triangles = target, diamonds = interferer.
          </p>
          <TimelineEditor />
        </div>

        {/* Right: Sources */}
        <div className="col-span-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-600">Sources</span>
            <button onClick={store.addSource}
              className="text-xs text-blue-600 hover:text-blue-800 font-medium">
              + Add
            </button>
          </div>
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
            {store.sources.map((_, i) => <SourceCard key={i} index={i} />)}
          </div>
        </div>
      </div>
    </section>
  );
}
