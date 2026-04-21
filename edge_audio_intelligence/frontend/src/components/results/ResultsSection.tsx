import { useState } from 'react';
import { useRunStore } from '../../stores/runStore';
import { getAudioUrl } from '../../api/client';
import { getRating, formatMetric, RATING_COLORS, METRIC_DISPLAY_NAMES } from '../../constants/ratings';
import type { RatingLevel } from '../../constants/ratings';
import Plot from 'react-plotly.js';

// ── Rating Badge ──
function RatingBadge({ rating }: { rating: RatingLevel | null }) {
  if (!rating) return <span className="text-slate-300">\u2014</span>;
  const colors = RATING_COLORS[rating];
  return (
    <span className={`inline-flex px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider border ${colors.bg} ${colors.text} ${colors.border}`}>
      {rating}
    </span>
  );
}

// ── Metrics Table ──
function MetricsTable({ metrics }: { metrics: Record<string, unknown> }) {
  const stages: Record<string, { key: string; value: number }[]> = {
    SSL: [], Beamforming: [], Enhancement: [], Diarization: [], ASR: [], System: [],
  };

  for (const [key, val] of Object.entries(metrics)) {
    if (val == null) continue;
    const v = val as number;
    if (key.includes('angular')) stages.SSL.push({ key, value: v });
    else if (key.includes('si_sdr')) stages.Beamforming.push({ key, value: v });
    else if (key === 'pesq' || key === 'stoi') stages.Enhancement.push({ key, value: v });
    else if (key === 'der') stages.Diarization.push({ key, value: v });
    else if (key.includes('wer') || key.includes('cer')) stages.ASR.push({ key, value: v });
    else stages.System.push({ key, value: v });
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50 border-b border-slate-200">
            <th className="px-3 py-2 text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider w-[20%]">Stage</th>
            <th className="px-3 py-2 text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider w-[30%]">Metric</th>
            <th className="px-3 py-2 text-right text-[11px] font-semibold text-slate-500 uppercase tracking-wider w-[25%]">Value</th>
            <th className="px-3 py-2 text-right text-[11px] font-semibold text-slate-500 uppercase tracking-wider w-[25%]">Rating</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(stages).map(([stageName, stageMetrics]) => {
            if (stageMetrics.length === 0) return null;
            return stageMetrics.map((m, i) => (
              <tr key={m.key} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                <td className="px-3 py-1.5 text-xs text-slate-500 font-medium">
                  {i === 0 ? stageName : ''}
                </td>
                <td className="px-3 py-1.5 text-xs text-slate-700">
                  {METRIC_DISPLAY_NAMES[m.key] || m.key}
                </td>
                <td className="px-3 py-1.5 text-right font-mono text-xs text-slate-800">
                  {formatMetric(m.key, m.value)}
                </td>
                <td className="px-3 py-1.5 text-right">
                  <RatingBadge rating={getRating(m.key, m.value)} />
                </td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Stage Result Card ──
function StageCard({ stage, plots }: {
  stage: { stage: string; method: string; latency_ms: number; metrics: Record<string, number> };
  plots: Record<string, unknown>;
}) {
  const [expanded, setExpanded] = useState(false);

  // Find relevant plots for this stage
  const plotMap: Record<string, string[]> = {
    'GCC-PHAT': ['input_waveform', 'spatial_spectrum'],
    'SRP-PHAT': ['input_waveform', 'spatial_spectrum'],
    'Delay-and-Sum': ['bf_waveform', 'bf_spectrogram', 'bf_scalogram'],
    'MVDR': ['bf_waveform', 'bf_spectrogram', 'bf_scalogram'],
    'Spectral-Subtraction': ['enh_waveform', 'enh_spectrogram', 'enh_scalogram'],
    'Wavelet-Enhancement': ['enh_waveform', 'enh_spectrogram', 'enh_scalogram'],
    'Spectral-Subtraction-gated': ['enh_waveform', 'enh_spectrogram'],
    'TEN-VAD': ['vad_overlay'],
    'AtomicVAD + TEN VAD': ['vad_overlay'],
    'Wavelet-VAD': ['vad_overlay'],
    'AtomicVAD': ['vad_overlay'],
    'Pyannote_3.1': ['diarization_timeline'],
    'Pyannote_3.0': ['diarization_timeline'],
  };

  const plotKeys = plotMap[stage.method] || plotMap[stage.stage] || [];
  const topMetrics = Object.entries(stage.metrics).slice(0, 3);

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden transition-all">
      <button onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-white hover:bg-slate-50/50 transition-colors text-left">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-sm text-slate-800">{stage.stage}</span>
          <span className="text-xs text-slate-400">{stage.method}</span>
          <span className="text-[10px] bg-slate-100 text-slate-500 rounded px-1.5 py-0.5 font-mono">
            {stage.latency_ms.toFixed(1)}ms
          </span>
        </div>
        <div className="flex items-center gap-2">
          {topMetrics.map(([k, v]) => (
            <span key={k} className="flex items-center gap-1">
              <span className="text-[10px] text-slate-400">{METRIC_DISPLAY_NAMES[k] || k}:</span>
              <span className="text-xs font-mono text-slate-700">{formatMetric(k, v)}</span>
              <RatingBadge rating={getRating(k, v)} />
            </span>
          ))}
          <svg className={`w-4 h-4 text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-slate-100 p-4">
          <div className="grid grid-cols-2 gap-3">
            {plotKeys.map(pk => {
              const plotData = plots[pk] as { data?: Plotly.Data[]; layout?: Partial<Plotly.Layout> } | undefined;
              if (!plotData?.data) return null;
              return (
                <div key={pk} className="bg-slate-50 rounded-lg p-1">
                  <Plot
                    data={plotData.data}
                    layout={{
                      ...plotData.layout,
                      autosize: true,
                      margin: { l: 40, r: 10, t: 25, b: 30 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                    }}
                    useResizeHandler
                    style={{ width: '100%', height: '200px' }}
                    config={{ displayModeBar: false, responsive: true }}
                  />
                </div>
              );
            })}
          </div>
          {plotKeys.every(pk => !plots[pk]) && (
            <p className="text-xs text-slate-400 text-center py-4">
              No visualizations available for this stage
            </p>
          )}

          {/* Full metrics for this stage */}
          {Object.keys(stage.metrics).length > 3 && (
            <div className="mt-3 pt-3 border-t border-slate-100">
              <div className="flex flex-wrap gap-2">
                {Object.entries(stage.metrics).map(([k, v]) => (
                  <span key={k} className="text-[10px] bg-white border border-slate-200 rounded px-2 py-1 text-slate-600">
                    <span className="text-slate-400">{METRIC_DISPLAY_NAMES[k] || k}:</span>{' '}
                    <span className="font-mono font-medium">{formatMetric(k, v)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Audio Signal Labels ──
function getSignalLabel(signal: string, result: { module_names?: Record<string, string> }): string {
  if (signal.startsWith('mic_')) return `Mic ${signal.split('_')[1]} (room mix)`;
  if (signal === 'beamformed') return `Beamformed (${result.module_names?.beamforming || 'BF'} output)`;
  if (signal === 'enhanced') return `Enhanced (${result.module_names?.enhancement || 'Enhancement'} output)`;
  if (signal.startsWith('clean_')) return `Clean Source ${signal.split('_')[1]} (dry, no room)`;
  return signal;
}

// ── Main Results Section ──
export function ResultsSection() {
  const { status, result } = useRunStore();
  const [selectedSignal, setSelectedSignal] = useState('mic_0');

  if (status !== 'complete' || !result) return null;

  const audioUrl = getAudioUrl(result.run_id, selectedSignal);

  return (
    <>
      {/* Output Panel */}
      <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-5 py-3 bg-emerald-50 border-b border-emerald-100 flex items-center gap-2">
          <div className="w-2 h-2 bg-emerald-500 rounded-full" />
          <h2 className="text-base font-semibold text-emerald-800">Pipeline Complete</h2>
          <span className="ml-auto text-xs font-mono text-emerald-600">
            {result.total_latency_ms.toFixed(0)}ms total | RTF {result.total_rtf.toFixed(3)}
          </span>
        </div>

        <div className="p-4 space-y-4">
          {/* Audio + Transcriptions */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block">
                <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wide">Audio Signal</span>
                <select value={selectedSignal} onChange={e => setSelectedSignal(e.target.value)}
                  className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-sm bg-white mt-0.5">
                  {result.audio_signals.map(s => (
                    <option key={s} value={s}>{getSignalLabel(s, result)}</option>
                  ))}
                </select>
              </label>
              <audio key={audioUrl} controls className="w-full mt-2 rounded-lg" src={audioUrl} />
            </div>

            <div>
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wide">Transcription</span>
              {result.transcriptions && result.transcriptions.length > 0 ? (
                <div className="space-y-1.5 mt-1">
                  {result.transcriptions.map((t, i) => (
                    <div key={i} className="bg-slate-50 rounded-lg p-2.5 border border-slate-100">
                      <div className="flex items-center gap-1.5 mb-1">
                        <span className="text-[10px] font-semibold text-white bg-emerald-600 rounded px-1.5 py-0.5">ASR</span>
                        <span className="text-[10px] text-slate-400">Source {i}</span>
                      </div>
                      <p className="text-sm text-slate-800 leading-relaxed">{t || '\u2014 (empty)'}</p>
                    </div>
                  ))}
                  {result.enhancement_gate_reason && (
                    <div className="text-[10px] text-slate-400 bg-slate-50 rounded px-2 py-1 border border-slate-100">
                      Enhancement: {result.enhancement_applied ? 'Applied' : 'Skipped'}{' '}
                      ({result.enhancement_gate_reason})
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-slate-400 mt-1">No ASR configured</p>
              )}
            </div>
          </div>

          {/* Metrics Table */}
          <MetricsTable metrics={result.metrics} />

          {/* Speaker Segments (Diarization) */}
          {result.speaker_segments && result.speaker_segments.length > 0 && (
            <div>
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wide">Speaker Segments</span>
              <div className="space-y-1 mt-1">
                {result.speaker_segments.map(([start, end, spk], i) => (
                  <div key={i} className="flex items-center gap-2 bg-slate-50 rounded-lg px-3 py-1.5 border border-slate-100">
                    <span className="text-[10px] font-semibold text-white rounded px-1.5 py-0.5"
                      style={{ backgroundColor: ['#E53935','#1E88E5','#43A047','#FB8C00'][i % 4] }}>
                      {spk}
                    </span>
                    <span className="text-xs text-slate-600 font-mono">
                      {(start as number).toFixed(2)}s — {(end as number).toFixed(2)}s
                    </span>
                    <span className="text-[10px] text-slate-400">
                      ({((end as number) - (start as number)).toFixed(1)}s)
                    </span>
                  </div>
                ))}
              </div>
              {result.diarization_method && (
                <p className="text-[10px] text-slate-400 mt-1">Method: {result.diarization_method}</p>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Stage Results */}
      <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-5 py-3 bg-slate-50 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-800">Stage Details</h2>
        </div>
        <div className="p-4 space-y-2">
          {result.stages.map((stage) => (
            <StageCard key={stage.stage} stage={stage} plots={result.plots} />
          ))}
        </div>
      </section>
    </>
  );
}
