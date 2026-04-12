import { usePipelineStore } from '../../stores/pipelineStore';
import { useSceneStore } from '../../stores/sceneStore';
import { useRunStore } from '../../stores/runStore';
import { startPipelineRun, connectPipelineWS } from '../../api/client';
import { PIPELINE_PRESETS } from '../../constants/presets';
import { SSL_OPTIONS, BF_OPTIONS, ENH_OPTIONS, ASR_OPTIONS, VAD_OPTIONS, DISPLAY_NAMES, STAGE_COLORS } from '../../constants/options';
import type { PipelineResult } from '../../types/pipeline';

function Select({ label, value, options, onChange }: {
  label: string; value: string;
  options: { label: string; value: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wide">{label}</span>
      <select value={value} onChange={e => onChange(e.target.value)}
        className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-sm bg-white mt-0.5
          focus:ring-2 focus:ring-blue-500 focus:border-blue-300 transition-all">
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

function FlowBadge({ label, color, isSkip }: { label: string; color: string; isSkip: boolean }) {
  if (isSkip) {
    return (
      <span className="px-2.5 py-1 rounded-full text-[11px] font-medium text-slate-400 border border-dashed border-slate-300 bg-slate-50">
        skip
      </span>
    );
  }
  return (
    <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold text-white shadow-sm"
      style={{ backgroundColor: color }}>
      {label}
    </span>
  );
}

function PipelineFlowDiagram() {
  const { ssl, bf, enh, asr, vad, enhGate } = usePipelineStore();

  const stages: { key: string; val: string; color: string; suffix?: string }[] = [];

  if (vad !== 'none') {
    stages.push({ key: 'vad', val: vad, color: STAGE_COLORS.vad });
  }
  stages.push({ key: 'ssl', val: ssl, color: STAGE_COLORS.ssl });
  stages.push({
    key: 'bf', val: bf, color: STAGE_COLORS.bf,
    suffix: bf === 'mvdr' && vad !== 'none' ? ' (\u03a6nn)' : '',
  });
  stages.push({
    key: 'enh', val: enh, color: STAGE_COLORS.enh,
    suffix: enhGate ? ' (gated)' : '',
  });
  stages.push({ key: 'asr', val: asr, color: STAGE_COLORS.asr });

  return (
    <div className="flex items-center justify-center gap-1 flex-wrap py-2">
      {stages.map((s, i) => (
        <div key={s.key} className="flex items-center gap-1">
          {i > 0 && <span className="text-slate-300 text-sm font-bold">\u2192</span>}
          <FlowBadge
            label={(DISPLAY_NAMES[s.val] || s.val) + (s.suffix || '')}
            color={s.color}
            isSkip={s.val === 'none'}
          />
        </div>
      ))}
    </div>
  );
}

export function PipelineSection() {
  const pipe = usePipelineStore();
  const sceneId = useSceneStore(s => s.sceneId);
  const sceneGenerated = useSceneStore(s => s.sceneGenerated);
  const run = useRunStore();

  const handleRun = async () => {
    if (!sceneId) return;
    try {
      const { task_id } = await startPipelineRun({
        scene_id: sceneId, ssl: pipe.ssl, bf: pipe.bf, enh: pipe.enh,
        asr: pipe.asr, vad: pipe.vad, enh_gate: pipe.enhGate,
      });
      run.setRunning(task_id);

      connectPipelineWS(task_id,
        (msg: unknown) => {
          const m = msg as Record<string, unknown>;
          if (m.stage === 'result' && m.data) {
            run.setComplete(m.data as PipelineResult);
          } else if (m.stage === 'error') {
            run.setError((m.message as string) || 'Pipeline failed');
          } else {
            run.addProgress({
              stage: m.stage as string, progress: m.progress as number,
              message: (m.message as string) || '', latency_ms: (m.latency_ms as number) || 0,
              metrics: (m.metrics as Record<string, number>) || {},
            });
          }
        },
      );
    } catch (err) {
      run.setError(String(err));
    }
  };

  const isRunning = run.status === 'running';
  const presetInfo = PIPELINE_PRESETS[pipe.preset];

  return (
    <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 bg-slate-50 border-b border-slate-200">
        <h2 className="text-base font-semibold text-slate-800">Pipeline Configuration</h2>
      </div>

      <div className="p-4 space-y-3">
        {/* Preset selector */}
        <div className="flex items-center gap-3">
          <div className="w-48">
            <Select label="Preset" value={pipe.preset}
              options={Object.entries(PIPELINE_PRESETS).map(([k, v]) => ({ label: v.label, value: k }))}
              onChange={pipe.applyPreset} />
          </div>
          {presetInfo && (
            <p className="text-xs text-slate-500 mt-4">{presetInfo.description}</p>
          )}
        </div>

        {/* Stage dropdowns */}
        <div className="grid grid-cols-5 gap-3">
          <Select label="SSL" value={pipe.ssl} options={SSL_OPTIONS} onChange={pipe.setSsl} />
          <Select label="Beamforming" value={pipe.bf} options={BF_OPTIONS} onChange={pipe.setBf} />
          <Select label="Enhancement" value={pipe.enh} options={ENH_OPTIONS} onChange={pipe.setEnh} />
          <Select label="ASR" value={pipe.asr} options={ASR_OPTIONS} onChange={pipe.setAsr} />
          <Select label="VAD" value={pipe.vad} options={VAD_OPTIONS} onChange={pipe.setVad} />
        </div>

        {/* Flow diagram */}
        <div className="bg-slate-50 rounded-lg border border-slate-100 px-3 py-1">
          <PipelineFlowDiagram />
        </div>

        {/* Enhancement gate + Run button */}
        <div className="flex items-center gap-4 pt-1">
          <label className="flex items-center gap-2 text-sm text-slate-600 select-none cursor-pointer">
            <input type="checkbox" checked={pipe.enhGate}
              onChange={e => pipe.setEnhGate(e.target.checked)}
              className="rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
            Skip enhancement if multi-speaker
          </label>

          <button onClick={handleRun} disabled={isRunning || !sceneGenerated}
            className={`ml-auto px-10 py-3 rounded-xl font-bold text-white text-sm tracking-wide transition-all
              ${isRunning ? 'bg-slate-400 cursor-wait' :
                sceneGenerated ? 'bg-emerald-600 hover:bg-emerald-700 shadow-md hover:shadow-lg active:scale-[0.98]' :
                'bg-slate-300 cursor-not-allowed'}`}>
            {isRunning ? 'RUNNING...' : 'RUN PIPELINE'}
          </button>
        </div>

        {/* Progress */}
        {isRunning && (
          <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 transition-all duration-500 ease-out rounded-full"
                style={{ width: `${run.progress * 100}%` }} />
            </div>
            <div className="flex justify-between mt-1.5">
              <span className="text-xs text-slate-500">{run.progressMessage}</span>
              <span className="text-xs text-slate-400 font-mono">{Math.round(run.progress * 100)}%</span>
            </div>
            {/* Per-stage latencies */}
            <div className="flex gap-2 mt-2 flex-wrap">
              {run.stages.filter(s => s.latency_ms > 0).map(s => (
                <span key={s.stage} className="text-[10px] bg-white rounded px-1.5 py-0.5 border border-slate-200 text-slate-600">
                  {s.stage}: {s.latency_ms.toFixed(0)}ms
                  {s.metrics.angular_error != null && ` (${s.metrics.angular_error.toFixed(1)}\u00b0)`}
                  {s.metrics.speech_pct != null && ` (${s.metrics.speech_pct.toFixed(0)}% speech)`}
                </span>
              ))}
            </div>
          </div>
        )}

        {run.status === 'error' && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
            {run.error}
          </div>
        )}
      </div>
    </section>
  );
}
