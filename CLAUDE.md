# CLAUDE.md — Master Project Intelligence File
# Edge Audio Intelligence System: Research & Development Guide
# Course: EE678 Wavelets and Multiresolution Signal Processing, IIT Bombay

> **This is the governing document for the entire project.**  
> Every coding session, every experiment, every decision references this file.  
> When in doubt about anything — architecture, naming, math, priorities — come here first.

---

## 0. PROJECT IDENTITY

**Title:** A Multi-Resolution, Energy-Efficient Edge Audio Intelligence System for Continuous Low-Power Streaming Speech Recognition in Real-World Environments

**Core Research Question:** Can a modular edge audio pipeline — integrating wavelet-based multi-resolution processing with a localization↔separation↔diarization feedback loop — achieve robust real-time speech transcription under diverse acoustic conditions while remaining deployable on embedded hardware?

**Course Alignment:** This is an EE678 (Wavelets) R&D project. Wavelet and multi-resolution concepts must appear substantively in at least three pipeline stages — not as decoration but as enabling components that provide measurable improvements in accuracy, interpretability, or efficiency.

**Deliverables:** (1) Working modular pipeline with benchmarks. (2) Research paper with ablation studies. (3) Acoustic simulation lab for reproducible evaluation. (4) This living documentation system.

---

## 1. COMPANION FILES — THE KNOWLEDGE SYSTEM

This project is governed by a set of interconnected .md files. Each has a specific role:

| File | Role | When to Read |
|------|------|--------------|
| `CLAUDE.md` (this file, project root) | Master rules, architecture, coding standards, workflow | Always. Before every session. |
| `docs/ACOUSTIC_LAB.md` | Simulation testbench design, scene generation, evaluation framework | When building/modifying the testbench |
| `docs/PIPELINE_ALGORITHM.md` | Every algorithm's math, design rationale, implementation notes per module | When implementing or debugging any module |
| `docs/V1_IMPLEMENTATION.md` | Exact spec for the first working version — what to build, in what order, with what tools | During initial build phase (Weeks 1-4) |
| `docs/RESEARCH_LOG.md` | Living document: discoveries, failures, surprises, benchmarks, decisions and their rationale | After every experiment. Before writing the paper. |
| `docs/DATASETS_AND_METRICS.md` | All datasets, download links, preprocessing, and every metric formula | When setting up data or evaluation |
| `docs/REFERENCES.md` | Curated bibliography with per-paper notes on what's useful and what's not | When writing the paper or looking for baselines |
| `docs/METHODOLOGY.md` | Research methodology: problem definition, constraints, metrics rationale, experimental roadmap, decision framework | Before any experiment. When choosing between algorithms. When writing the paper's methodology section. |

**Rule: When any experiment reveals new information — a bottleneck, a surprising result, a failed approach, a parameter sensitivity — it MUST be logged in `RESEARCH_LOG.md` immediately. This file is the raw material for the research paper.**

---

## 2. PHILOSOPHY AND PRINCIPLES

### 2.1 Build-First, Not Read-First

The research methodology is iterative:
```
Build baseline → Measure everything → Identify bottleneck → 
Read targeted papers → Improve one module → Re-measure → Repeat
```
Never spend more than 2 days reading without running code. The pipeline teaches you what matters faster than papers do.

### 2.2 One Module at a Time

Never modify two modules simultaneously. When you change the SSL algorithm, keep everything else constant. When you upgrade the enhancer, keep the same SSL and beamformer. This is the only way to produce valid ablation studies for the paper.

### 2.3 Measure Before and After Everything

Every module change must produce a row in a comparison table. If you can't measure the effect, you can't publish it. The minimum measurement for any change is: (a) the module-level metric (e.g., RMSAE for SSL), (b) the end-to-end WER, (c) the latency impact.

### 2.4 Simulation First, Real Data Second

Train and develop entirely on simulated data (pyroomacoustics + LibriSpeech + MUSAN). Only test on real data (LOCATA, AMI) at the end for validation. This ensures reproducibility and controlled experimentation.

### 2.5 Wavelet Integration Must Be Substantive

Do not add wavelets as a post-hoc wrapper. They must be integrated into the signal flow at points where multi-resolution analysis provides a genuine advantage:
- **Enhancement:** DWT sub-band features replace or augment STFT, providing selective frequency-band denoising.
- **SSL:** Wavelet-initialized convolution kernels in the multi-stream CNN, providing interpretable multi-scale delay analysis.
- **VAD/always-on listener:** Wavelet energy features for ultra-low-compute speech activity detection.
- **Interpretability:** Wavelet coefficient visualization at each pipeline stage to explain what information flows where.

---

## 3. CODE ARCHITECTURE

### 3.1 Directory Structure

> **Status:** V1 complete. Items marked [V2] are planned but not yet implemented.

```
edge_audio_intelligence/
│
├── algorithms/                    # PURE MATH — no pipeline/dashboard dependencies
│   ├── rir.py                     # RIR generation (ISM), convolution, Sabine
│   ├── signal_mixing.py           # Multi-source mixing, SNR-calibrated noise
│   └── doa.py                     # DOA computation (azimuth, elevation, TDOA)
│
├── modules/                       # PIPELINE MODULES — plug-and-play
│   ├── base.py                    # BaseModule abstract class
│   ├── ssl/                       # Sound Source Localization
│   │   ├── base.py                # BaseSSL interface
│   │   ├── gcc_phat.py            # [IMPLEMENTED] GCC-PHAT
│   │   ├── srp_phat.py            # [IMPLEMENTED] SRP-PHAT
│   │   └── music.py               # [IMPLEMENTED] MUSIC
│   ├── beamforming/
│   │   ├── base.py
│   │   ├── delay_and_sum.py       # [IMPLEMENTED] Delay-and-Sum
│   │   └── mvdr.py                # [IMPLEMENTED] MVDR
│   ├── enhancement/
│   │   ├── base.py
│   │   ├── spectral_subtraction.py # [IMPLEMENTED] Classical spectral subtraction
│   │   └── wavelet_enhancement.py  # [IMPLEMENTED] DWT sub-band denoising
│   ├── asr/
│   │   ├── base.py
│   │   └── whisper_offline.py      # [IMPLEMENTED] Whisper tiny/base/small
│   ├── separation/                 # [V2] Speaker separation
│   │   └── base.py                 # Interface only
│   └── diarization/                # [V2] Speaker diarization
│       └── base.py                 # Interface only
│
├── wavelet/                       # Wavelet utilities (EE678 course-specific)
│   ├── dwt_features.py            # DWT decomposition, sub-band energy features
│   ├── wavelet_vad.py             # Wavelet energy-based VAD
│   ├── wavelet_init.py            # Wavelet-initialized conv kernels
│   └── analysis.py                # Scalogram, stage comparison, interpretability
│
├── testbench/                     # Simulation orchestration (uses algorithms/)
│   ├── scene.py                   # SceneConfig, AcousticScene, MicArrayConfig
│   ├── simulator.py               # RIRSimulator — calls algorithms/ to build scenes
│   ├── evaluator.py               # PipelineEvaluator — metrics at every probe point
│   └── visualizer.py              # Plotting utilities
│
├── pipeline/                      # Pipeline orchestration
│   ├── cascade.py                 # [IMPLEMENTED] Level 1: sequential pipeline
│   └── runner.py                  # ExperimentRunner with logging
│
├── dashboard/                     # Interactive Dash web UI
│   ├── app.py                     # Dash app factory (run on port 8050)
│   ├── layout.py                  # Full page layout
│   ├── state.py                   # Run history persistence + saved setups
│   ├── components/                # UI components (scene, pipeline, results, history)
│   ├── callbacks/                 # Event handlers (scene, run, audio, history, setup)
│   └── assets/                    # CSS, icons
│
├── experiments/                   # Experiment scripts
│   ├── 01_baseline_pipeline.py    # [DONE] V1 baseline on corner cases
│   └── 02_full_pipeline_with_asr.py # [DONE] Full pipeline with Whisper
│
├── utils/
│   ├── audio_io.py                # Load, save, resample audio
│   ├── metrics.py                 # All metrics (WER, CER, PESQ, STOI, SI-SDR, etc.)
│   ├── visualization.py           # Matplotlib plotting helpers
│   └── profiling.py               # Latency, memory measurement
│
├── tests/                         # Unit tests
│   ├── test_ssl.py, test_beamforming.py, test_enhancement.py
│   ├── test_pipeline.py, test_wavelet.py
│   └── conftest.py
│
├── config/scenes/                 # YAML scene presets
├── results/                       # Auto-generated: tables/, figures/, logs/, audio/
│   └── run_history.json           # Persistent run history from dashboard
│
├── data/                          # Datasets (at project root, not committed)
│   └── librispeech/               # LibriSpeech test-clean (2620 utterances)
│
└── requirements.txt
```

### 3.2 The BaseModule Contract

Every module must implement this interface. No exceptions.

```python
class BaseModule(ABC):
    """Abstract base class for all pipeline modules."""
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        The input and output are ALWAYS dictionaries. Keys are standardized
        per module type (see Section 3.3). This enables plug-and-play swapping.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return all configuration parameters. Used for experiment logging."""
        pass
    
    def measure_latency(self, data: Dict, n_runs: int = 100) -> float:
        """Measure average processing latency in milliseconds."""
        pass
    
    def count_parameters(self) -> int:
        """Return total trainable parameters (0 for non-neural methods)."""
        pass
    
    def estimate_flops(self, data: Dict) -> int:
        """Estimate FLOPs for one forward pass."""
        pass
```

### 3.3 Standardized Data Flow Between Modules

The data flowing between modules is ALWAYS a dictionary with standardized keys. This is critical for plug-and-play.

```
SCENE INPUT → {
    "multichannel_audio": np.ndarray [n_mics, n_samples],
    "sample_rate": int,
    "mic_positions": np.ndarray [n_mics, 3],
    "ground_truth": {  # Only available in simulation
        "doa_per_source": list of np.ndarray [n_frames, 2],  # azimuth, elevation
        "clean_sources": list of np.ndarray [n_samples],
        "transcriptions": list of str,
        "speaker_labels": list of (start, end, speaker_id),
        "n_sources": int
    }
}

SSL OUTPUT → adds {
    "estimated_doa": np.ndarray [n_frames, n_sources, 2],
    "doa_confidence": np.ndarray [n_frames, n_sources],
    "n_detected_sources": int,
    "ssl_method": str,
    "ssl_latency_ms": float
}

BEAMFORMING OUTPUT → adds {
    "beamformed_audio": np.ndarray [n_samples] or [n_sources, n_samples],
    "spatial_spectrum": np.ndarray [n_directions],  # optional
    "bf_method": str,
    "bf_latency_ms": float
}

ENHANCEMENT OUTPUT → adds {
    "enhanced_audio": np.ndarray [n_samples] or [n_sources, n_samples],
    "enhancement_method": str,
    "enhancement_latency_ms": float
}

SEPARATION OUTPUT → adds {
    "separated_sources": list of np.ndarray [n_samples],
    "separation_method": str,
    "separation_latency_ms": float
}

ASR OUTPUT → adds {
    "transcriptions": list of str,
    "word_timestamps": list of list of (word, start, end),
    "asr_method": str,
    "asr_latency_ms": float
}

DIARIZATION OUTPUT → adds {
    "speaker_segments": list of (start, end, speaker_id),
    "attributed_transcription": list of (speaker_id, text, start, end),
    "diarization_method": str,
    "diarization_latency_ms": float
}
```

### 3.4 Coding Standards

1. **Every function has a docstring** explaining inputs, outputs, and the mathematical operation performed. Reference the equation number from `PIPELINE_ALGORITHMS.md`.

2. **No magic numbers.** Every constant comes from config or is a named variable with a comment citing its source (paper, table number, etc.).

3. **Type hints everywhere.** Use `np.ndarray`, `torch.Tensor`, `Dict[str, Any]`, etc.

4. **Unit tests for math.** Before implementing any algorithm, write a test with a known-answer case. For GCC-PHAT: two delayed sine waves → verify the detected delay matches the true delay. For MVDR: single source in noise → verify the output SNR improvement matches the theoretical prediction.

5. **Logging.** Every experiment auto-logs: config, all metric values, timestamps, git commit hash. Use Python logging module, not print statements.

6. **Reproducibility.** Every random operation uses a seeded generator. The seed is part of the config and logged.

7. **Separation of concerns.** Signal processing code never imports visualization code. Experiment scripts never contain algorithm implementations. Configs are YAML, never hardcoded.

---

## 4. EXPERIMENT WORKFLOW

### 4.1 For Every Experiment

```
1. Define hypothesis: "Replacing spectral subtraction with DWT-based 
   enhancement will improve PESQ by >0.3 without increasing latency."

2. Write config YAML specifying all parameters.

3. Run experiment script. It automatically:
   a. Generates or loads acoustic scenes
   b. Runs the pipeline
   c. Computes all metrics at every probe point
   d. Saves results to results/ directory
   e. Appends summary to RESEARCH_LOG.md

4. Analyze results. Update RESEARCH_LOG.md with:
   - The numbers (in a table)
   - Whether hypothesis was confirmed or rejected
   - Why (your interpretation)
   - What to try next

5. If the result changes architecture decisions, update CLAUDE.md.
```

### 4.2 The SNR×RT60 Grid

Every module-level comparison must be evaluated across this standardized grid:

```
SNR levels:  [5, 15, 30] dB
RT60 levels: [0.0, 0.3, 0.6, 0.9, 1.2, 1.5] s

Corner cases (for quick checks):
  LL: SNR=30 dB, RT60=0.0s  (easy, sanity check)
  HL: SNR=5 dB,  RT60=0.0s  (noisy, no reverb)
  LH: SNR=30 dB, RT60=1.5s  (clean, high reverb)
  HH: SNR=5 dB,  RT60=1.5s  (hard, stress test)
```

This follows the Cross3D-Edge paper's evaluation methodology (Yin & Verhelst, 2025, Section 4.3) and ensures fair comparison across all conditions.

### 4.3 The Ablation Protocol

When testing module X with variants A, B, C:

1. Fix ALL other modules at their current best version.
2. Run variant A on the full SNR×RT60 grid. Record module-level metric AND end-to-end WER.
3. Repeat for B, C.
4. Present results in two tables: (a) module-level metric grid, (b) end-to-end WER grid.
5. Report latency and parameter count for each variant.
6. Select winner. Document reasoning in RESEARCH_LOG.md.
7. Lock in the winner. Move to next module.

---

## 5. WAVELET INTEGRATION STRATEGY

This section defines exactly how wavelet concepts connect to the project. This is critical for course alignment.

### 5.1 Enhancement: DWT Sub-band Processing

**Where:** `modules/enhancement/wavelet_enhancement.py` and `wavelet/dwt_features.py`

**What:** Apply J-level DWT (J=3, CDF 5/3 wavelet) to decompose each frame into sub-bands. Apply level-dependent soft thresholding for denoising. Optionally feed sub-band features into a lightweight CNN for learned thresholding.

**Why this works:** Speech energy concentrates in the approximation coefficients (low frequency), while broadband noise distributes uniformly across all sub-bands. The wavelet transform provides natural frequency-dependent noise estimation.

**What to measure:** PESQ, STOI, SI-SDR with and without DWT features. Visualize the sub-band energy distributions for speech vs. noise to demonstrate interpretability.

### 5.2 SSL: Wavelet-Initialized Convolution Filters

**Where:** `wavelet/wavelet_init.py`, used by `modules/ssl/multi_stream_cnn.py`

**What:** Initialize the parallel convolution kernels (sizes 3, 5, 7) in the multi-stream block using Daubechies wavelet filter coefficients (db2 for k=3, db3 for k=5, db4 for k=7). Compare convergence speed and final accuracy vs. random initialization.

**Why this works:** Wavelet filters are optimal for capturing features at specific scales. The multi-stream CNN is already performing multi-scale analysis — wavelet initialization makes this explicit and well-conditioned.

**What to measure:** Training loss curves (convergence speed), final RMSAE, and intermediate feature visualizations showing what each kernel detects.

### 5.3 Always-On Listener: Wavelet Energy VAD

**Where:** `wavelet/wavelet_vad.py`

**What:** Compute per-frame energy in each DWT sub-band. Speech has high energy in cA_3 (0-1kHz) and cD_3 (1-2kHz) relative to cD_1 (4-8kHz). A simple threshold on the ratio E(cA_3)/E(cD_1) provides effective VAD with minimal computation — just the DWT (a few multiply-accumulates per sample) and energy computation.

**Why this matters:** This is the ultra-low-power front-end that decides whether to wake up heavier models. It must run at <5mW. Wavelet energy is far cheaper than computing an STFT.

**What to measure:** VAD accuracy (precision, recall, F1) compared to energy-based VAD and neural VAD. Also compute FLOPS per frame.

### 5.4 Interpretability Analysis

**Where:** `wavelet/analysis.py`

**What:** At each pipeline stage, compute and visualize the wavelet scalogram of the signal. Show how SSL extracts delay information from different sub-bands. Show how enhancement suppresses noise sub-bands while preserving speech sub-bands. Show how separation disentangles speakers across time-frequency scales.

**Why this matters:** This provides the interpretability argument in the paper — we can explain what the system does at each stage in terms of multi-resolution signal structure, not just as a black box neural network.

---

## 6. THE FEEDBACK LOOP — SYSTEM-LEVEL CONTRIBUTION

This is the most publishable aspect of the project. Three levels of integration:

### Level 1: Cascade (Baseline)
```
SSL → Beamforming → Enhancement → Separation → ASR → Diarization
(each module independent, no information flows backward)
```

### Level 2: Guided (Target)
```
SSL → Beamforming → Enhancement → Separation → ASR → Diarization
 │                                    ↑                        │
 └──── DoA spatial features ──────────┘                        │
 └──── Speaker embeddings for tracking ────────────────────────┘
```
SSL DoA estimates are injected as spatial features (IPD, ILD) into the separation module. Diarization speaker embeddings are fed back to SSL for source tracking across frames.

### Level 3: Joint (Stretch Goal)
End-to-end fine-tuning of separation + ASR with a joint CTC/attention loss. The separation front-end and ASR back-end share gradients. This requires careful implementation but produces the best WER.

**For the paper:** Compare all three levels. The delta between Level 1 and Level 2 is the key result. Level 3 is bonus.

---

## 7. KEY DECISIONS LOG

| Date | Decision | Rationale | Alternative Considered |
|------|----------|-----------|----------------------|
| 2026-03-17 | Use pyroomacoustics ISM for RIR simulation | Well-documented API, ISM gives accurate RIRs for shoebox rooms, built-in DOA algorithms | gpuRIR (faster but less integrated) |
| 2026-03-17 | CDF 5/3 wavelet (bior2.2) for enhancement | Low compute, symmetric, good frequency selectivity, used in JPEG2000 | Daubechies db4 (higher order but more compute) |
| 2026-03-18 | Whisper tiny/base/small for V1 ASR (not Emformer) | Pre-trained, no training needed, 3 sizes enable cascaded inference ablation | Emformer (requires training, V2 target) |
| 2026-03-19 | Separated algorithms/ package from testbench/ | Pure math functions (RIR, mixing, DOA) can be imported and used anywhere without simulation dependencies | Keep all math inside simulator (less modular) |
| 2026-03-19 | mm-scale mic spacing (14-58mm) for edge realism | Real phones have 14mm spacing, ReSpeaker has 58mm diameter. Previous 5-8cm was unrealistic for edge devices | 5cm (too large for real devices) |
| 2026-03-19 | RT60 < 0.2s treated as anechoic | pyroomacoustics inverse_sabine fails for small RT60 + normal room sizes. Anechoic is equivalent for very low RT60 | Crash on small RT60 |
| 2026-03-20 | Interactive Dash dashboard for experiment management | Enables rapid A/B testing: change one module, run, compare. Saves run history with remarks for paper. | CLI-only experiments (slower iteration) |
| 2026-03-22 | GCC-PHAT 16x upsampling for mm-scale arrays | Max TDOA at 15mm/16kHz is 2.1 samples — integer resolution gives only 1 DOA bin. 16x upsampling reduces error from 18° to 1.7° | Parabolic interpolation (insufficient for <3 sample TDOA) |
| 2026-03-22 | Three-method VAD (AtomicVAD + TEN VAD + Wavelet) | Different constraints: always-on wake (<5mW) vs precision timestamps (MVDR Phi_nn) vs EE678 baseline | Single VAD method (loses specialization and ablation story) |
| 2026-03-22 | MVDR Phi_nn gating on VAD noise labels | +13.8dB SI-SDR improvement. Without gating, speech contaminates noise covariance and MVDR suppresses target | Ungated MVDR (V1 approach — proven to fail) |
| 2026-03-22 | Enhancement gate (skip when n_speakers > 1) | WER recovery 39.3% → 25% in multi-speaker. Enhancement treats interfering speech as noise | Always-on enhancement (V1 approach — 14pp WER degradation) |
| 2026-03-22 | Wavelet threshold scaling follows speech spectral tilt | Fine details (4-8kHz) get larger thresholds; coarse details (0-2kHz) get smaller. Matches -6dB/octave speech spectrum | Inverted scaling (was preserving noise, removing speech) |
| 2026-03-22 | Model caching for TF and Whisper | 10.9x pipeline speedup (6.9s → 0.6s). First run loads, subsequent runs reuse | Reload every run (3.4s TF + 2.5s Whisper overhead per run) |
| 2026-03-22 | Tonal interference detection in GatedEnhancer | Detects spectral peaks >15dB above floor, skips enhancement. WER 40.9% → 27.3%. Whisper handles tones better than processing artifacts | Apply enhancement always (V1 — adds 14pp WER) |
| 2026-03-22 | DS preferred over MVDR for tonal interference | MVDR degrades WER by +45pp despite +5dB SI-SDR. 15mm array aperture (λ/7.6 at 1kHz) too small for spatial filtering without speech distortion | MVDR for all noise types (wrong — MVDR only for broadband) |

---

## 8. CRITICAL REMINDERS

1. **Never commit a change without measuring its effect.** The results/ directory must grow with every code change.

2. **The paper writes itself from RESEARCH_LOG.md.** Every table, every figure, every insight in the paper should trace back to a logged experiment.

3. **Edge deployment is a constraint, not an afterthought.** Every algorithm choice must report latency and FLOPS alongside accuracy. If a model achieves 2% better WER but 10x more latency, that's usually not worth it for this project.

4. **The testbench is your most valuable asset.** Invest time in making it bulletproof. If the testbench is correct, every experiment is trustworthy. If the testbench has bugs, nothing you measure means anything.

5. **Update this file.** When a principle changes, when a new companion file is created, when the architecture evolves -- update CLAUDE.md. This file is the single source of truth.

---

## 9. V1 RESULTS SUMMARY (Completed 2026-03-20)

### V1 Pipeline: GCC-PHAT / MUSIC -> Delay-and-Sum -> Spectral Sub / Wavelet -> Whisper

**Baseline on synthetic signals (corner cases, sine wave source):**

| Condition | SSL Error | BF SI-SDR | Enh PESQ | WER | Total Latency |
|-----------|-----------|-----------|----------|-----|---------------|
| LL (SNR=30, RT60=0.0) | 7.7 deg | -13.9 dB | 1.20 | -- | 7.2ms |
| HL (SNR=5, RT60=0.0) | 9.1 deg | -15.8 dB | 1.01 | -- | 5.2ms |
| LH (SNR=30, RT60=1.5) | 40.2 deg | 12.5 dB | 1.18 | -- | 5.1ms |
| HH (SNR=5, RT60=1.5) | 36.2 deg | 8.6 dB | 1.01 | -- | 4.9ms |

**With LibriSpeech speech + Whisper (dashboard runs, Office Meeting preset):**

| Pipeline | SSL Error | PESQ | STOI | WER | CER | Latency |
|----------|-----------|------|------|-----|-----|---------|
| GCC-PHAT -> DS -> SpectSub -> Whisper-small | 80.8 deg | 1.05 | 0.570 | 110.7% | -- | 3058ms |
| GCC-PHAT -> DS -> SpectSub -> Whisper-small (clean room) | 3.9 deg | 1.92 | 0.879 | 21.4% | -- | 4566ms |
| MUSIC -> DS -> Wavelet -> Whisper-small (office) | 48.0 deg | 1.10 | 0.643 | 39.3% | 20.9% | 2995ms |
| MUSIC -> DS (no enh) -> Whisper-small (office) | 48.0 deg | -- | -- | 25.0% | -- | 3022ms |

### Key Bottlenecks Identified

1. **SSL accuracy is POOR in reverberant/multi-source conditions** (40-80 deg error vs target <10 deg). GCC-PHAT degrades heavily with reflections. MUSIC slightly better but still far from acceptable. This directly hurts beamforming which depends on accurate DOA.

2. **Enhancement PESQ is POOR** (1.0-1.9 range vs target >2.5). Both spectral subtraction and wavelet enhancement produce low-quality output. The enhancer may actually be degrading the signal -- removing WER is better without enhancement in some cases (25% WER without vs 39% with Wavelet in office scene).

3. **ASR dominates latency** (Whisper-small: ~3000ms of ~3050ms total). Signal processing modules (SSL + BF + Enhancement) together take <60ms. ASR is 98% of total latency. For edge deployment, Whisper-tiny (0.3s) or a streaming model is essential.

4. **Multi-source scenario is unsolved**: no separation module means interfering speaker corrupts the target signal. WER degrades from 21% (clean room, single speaker) to 39-110% (multi-speaker).

---

## 10. V2 ROADMAP -- IMPROVING RESULTS

### COMPLETED V2 Improvements (2026-03-22)

**SSL Fix (Priority 1): DONE**
- GCC-PHAT 16x upsampling: error reduced from 18° → 1.7° (clean), 40° → 9.5° (RT60=0.4)
- Meets <10° target for RT60 ≤ 0.4s. Still exceeds target at RT60 ≥ 0.6s (14-20°).

**VAD Integration: DONE**
- Three-method VAD: AtomicVAD (wake), TEN VAD (Phi_nn), Wavelet (baseline)
- MVDR Phi_nn gating: +13.8dB SI-SDR improvement
- Enhancement gate: recovers ~14pp WER in multi-speaker

**Algorithm Bug Fixes: DONE**
- DS fractional delay sign error fixed (+5.7dB improvement)
- Wavelet threshold scaling inverted (now matches speech spectral tilt)
- Spectral subtraction uses VAD-guided noise estimation
- Evaluator SI-SDR reference and active-region comparison fixed
- Model caching: 10.9x pipeline speedup

### Remaining V2 Priorities:

### Priority 1: SSL in high reverberation (Current: 14-20 deg at RT60≥0.6 -> Target: <10 deg)

### Priority 2: Improve enhancement quality (Current: PESQ 1.0-1.9 -> Target: >2.5)
- **Approach:** Tune wavelet enhancement (vary wavelet type, levels, threshold). Run ablation: bior2.2 vs db4 vs sym4, levels 2-5, threshold scales 0.5-2.0.
- **If still poor:** Implement DeepFilterNet (neural enhancement).
- **Measurement:** PESQ, STOI, SI-SDR. Also measure WER with and without enhancement to check if enhancement helps or hurts ASR.

### Priority 3: Reduce ASR latency (Current: ~3000ms -> Target: <500ms)
- **Approach:** Use Whisper-tiny (39M params, 0.3s) as default for edge. Reserve Whisper-small for accuracy upper bound comparison.
- **For streaming:** Implement Emformer (V2 stretch).
- **Measurement:** WER vs latency tradeoff across tiny/base/small.

### Priority 4: Multi-source handling (V2)
- **Approach:** Implement Conv-TasNet for 2-speaker separation.
- **Prerequisite:** SSL must be accurate enough to provide spatial cues.
- **Measurement:** SI-SDRi on WSJ0-2mix or LibriMix.

### Priority 5: Speaker diarization (V2)
- **Approach:** Integrate pyannote.audio for who-spoke-when attribution.
- **Measurement:** DER on AMI Meeting Corpus.

**For every improvement: follow the Ablation Protocol (Section 4.3). Change ONE module. Measure. Log in RESEARCH_LOG.md. Repeat.**

---

*Last updated: 2026-03-20*
*Next review: After V2 SSL improvement experiments*