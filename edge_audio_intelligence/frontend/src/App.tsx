import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SceneSection } from './components/scene/SceneSection';
import { PipelineSection } from './components/pipeline/PipelineSection';
import { ResultsSection } from './components/results/ResultsSection';
import { HistorySection } from './components/history/HistorySection';

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-slate-50">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <h1 className="text-2xl font-bold text-slate-800">
              Edge Audio Intelligence Lab
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Acoustic simulation &amp; pipeline evaluation
            </p>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
          {/* Section 1: Scene Setup */}
          <SceneSection />

          {/* Section 2: Pipeline Configuration */}
          <PipelineSection />

          {/* Section 3+4: Results (appears after run) */}
          <ResultsSection />

          {/* Section 5: Run History */}
          <HistorySection />
        </main>
      </div>
    </QueryClientProvider>
  );
}
