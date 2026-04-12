import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getHistory, deleteRun } from '../../api/client';
import { getRating, formatMetric, RATING_COLORS, METRIC_DISPLAY_NAMES } from '../../constants/ratings';

interface HistoryRun {
  run_id: string;
  snr_db: number;
  rt60: number;
  n_sources: number;
  total_latency_ms: number;
  total_rtf: number;
  end_to_end_wer: number | null;
  module_results: { name: string; metrics: Record<string, number>; latency_ms: number }[];
}

function HistoryCard({ run }: { run: HistoryRun }) {
  const [expanded, setExpanded] = useState(false);
  const queryClient = useQueryClient();

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Delete run ${run.run_id}?`)) {
      await deleteRun(run.run_id);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    }
  };

  // Build pipeline summary
  const pipelineSummary = run.module_results.map(mr => mr.name).join(' \u2192 ');

  // Key metrics
  const allMetrics: Record<string, number> = {};
  for (const mr of run.module_results) {
    Object.assign(allMetrics, mr.metrics);
  }

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden transition-all hover:border-slate-300">
      <button onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-2.5 bg-white hover:bg-slate-50/30 text-left transition-colors">
        {/* Run ID */}
        <span className="font-mono text-xs text-slate-400 w-16 shrink-0">
          #{run.run_id.slice(-4)}
        </span>

        {/* Pipeline summary */}
        <span className="text-xs text-slate-600 truncate flex-1">
          {pipelineSummary || 'No modules'}
        </span>

        {/* Scene info */}
        <span className="text-[10px] bg-slate-100 text-slate-500 rounded px-1.5 py-0.5 shrink-0">
          SNR={run.snr_db}dB RT60={run.rt60}s {run.n_sources}src
        </span>

        {/* Key metrics inline */}
        {allMetrics.angular_error_deg != null && (
          <span className="text-[10px] text-slate-600 shrink-0">
            SSL: {allMetrics.angular_error_deg.toFixed(1)}\u00b0
          </span>
        )}
        {allMetrics.si_sdr_db != null && (
          <span className="text-[10px] text-slate-600 shrink-0">
            SDR: {allMetrics.si_sdr_db.toFixed(1)}dB
          </span>
        )}

        {/* Latency */}
        <span className="text-[10px] font-mono text-slate-400 shrink-0">
          {run.total_latency_ms.toFixed(0)}ms
        </span>

        {/* Actions */}
        <button onClick={handleDelete}
          className="text-slate-300 hover:text-red-500 transition-colors shrink-0 p-0.5"
          title="Delete run">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <svg className={`w-4 h-4 text-slate-400 transition-transform shrink-0 ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="border-t border-slate-100 p-4 bg-slate-50/30">
          {/* Per-module breakdown */}
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50">
                  <th className="px-3 py-1.5 text-left text-[10px] font-semibold text-slate-500 uppercase">Module</th>
                  <th className="px-3 py-1.5 text-left text-[10px] font-semibold text-slate-500 uppercase">Metric</th>
                  <th className="px-3 py-1.5 text-right text-[10px] font-semibold text-slate-500 uppercase">Value</th>
                  <th className="px-3 py-1.5 text-right text-[10px] font-semibold text-slate-500 uppercase">Rating</th>
                  <th className="px-3 py-1.5 text-right text-[10px] font-semibold text-slate-500 uppercase">Latency</th>
                </tr>
              </thead>
              <tbody>
                {run.module_results.map((mr) =>
                  Object.entries(mr.metrics).map(([k, v], mi) => (
                    <tr key={`${mr.name}-${k}`} className="border-t border-slate-100">
                      <td className="px-3 py-1 text-slate-600 font-medium">
                        {mi === 0 ? mr.name : ''}
                      </td>
                      <td className="px-3 py-1 text-slate-500">
                        {METRIC_DISPLAY_NAMES[k] || k}
                      </td>
                      <td className="px-3 py-1 text-right font-mono text-slate-700">
                        {v != null ? formatMetric(k, v) : '\u2014'}
                      </td>
                      <td className="px-3 py-1 text-right">
                        {v != null && getRating(k, v) && (
                          <span className={`inline-flex px-1.5 py-0.5 rounded text-[9px] font-semibold
                            ${RATING_COLORS[getRating(k, v)!].bg} ${RATING_COLORS[getRating(k, v)!].text}`}>
                            {getRating(k, v)}
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-1 text-right font-mono text-slate-400">
                        {mi === 0 ? `${mr.latency_ms.toFixed(1)}ms` : ''}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* System metrics */}
          <div className="flex gap-3 mt-3 text-[10px] text-slate-500">
            <span>Total: <span className="font-mono font-medium">{run.total_latency_ms.toFixed(0)}ms</span></span>
            <span>RTF: <span className="font-mono font-medium">{run.total_rtf.toFixed(3)}</span></span>
            {run.end_to_end_wer != null && (
              <span>WER: <span className="font-mono font-medium">{(run.end_to_end_wer * 100).toFixed(1)}%</span></span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function HistorySection() {
  const { data: history, isLoading } = useQuery({
    queryKey: ['history'],
    queryFn: getHistory,
    refetchInterval: 10000,
  });

  const runs = (history || []) as HistoryRun[];

  return (
    <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-800">Run History</h2>
        {runs.length > 0 && (
          <span className="text-xs text-slate-400">{runs.length} run(s)</span>
        )}
      </div>

      <div className="p-4">
        {isLoading ? (
          <p className="text-sm text-slate-400 text-center py-4">Loading history...</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-slate-400 text-center py-4">
            No runs yet. Configure a scene and run the pipeline.
          </p>
        ) : (
          <div className="space-y-2">
            {runs.map(run => <HistoryCard key={run.run_id} run={run} />)}
          </div>
        )}
      </div>
    </section>
  );
}
