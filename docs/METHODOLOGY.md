# METHODOLOGY.md — Research Methodology and System Design from First Principles
# Edge Audio Intelligence System: EE678 Wavelets & Multiresolution Signal Processing
# Authors: Anmol Sureka (24B2470)
# Last updated: 2026-03-20

---

## 0. WHY THIS DOCUMENT EXISTS

This document answers one question: **How do we systematically build an edge audio pipeline that produces state-of-the-art results, and how do we PROVE that each design choice is correct?**

We do not guess. We do not randomly swap modules. Every decision follows from:
1. A clearly stated problem
2. Physical/mathematical constraints that limit solutions
3. Evidence from experiments or published research
4. Measurable criteria for success or failure

---

## 1. THE PROBLEM — STATED PRECISELY

### 1.1 What we are building

A system that takes **raw multichannel audio from a small microphone array** (2-4 mics, 14-58mm spacing) and produces:
- **Who** is speaking (speaker identity/diarization)
- **What** they said (transcription)
- **Where** they are (direction of arrival)
- In **real time** (<500ms end-to-end latency)
- On **constrained hardware** (<50M parameters, <5W power)

### 1.2 Why this is hard — the fundamental challenges

**Challenge 1: The Cocktail Party Problem**
When multiple people speak simultaneously in a room, the microphone captures a mixture. Separating individual speakers from a mixture is an ill-posed problem — there are infinitely many combinations of source signals that could produce the same mixture.

*Why classical methods fail:* Spectral subtraction and wavelet thresholding assume noise is stationary and separable from speech in the frequency domain. But a second speaker's voice occupies the SAME frequency bands as the target speaker. You cannot separate them by frequency alone — you need spatial information.

**Challenge 2: Reverberation**
Sound bounces off walls, ceiling, and floor. By the time it reaches the microphone, the signal is smeared over time. A 0.4s RT60 room creates echoes that persist for 400ms — longer than most phonemes (50-200ms). This means the room itself distorts the speech, making both localization and recognition harder.

*The math:* The received signal is `x(t) = s(t) * h(t) + n(t)` where `h(t)` is the room impulse response with hundreds of delayed, attenuated copies of the original signal. For RT60=0.4s at 16kHz, the RIR has ~6400 non-zero samples. The convolution smears temporal features that ASR depends on.

**Challenge 3: Latency vs Accuracy Tradeoff**
The best speech recognition model (Whisper-large, 1.5B params) achieves <3% WER but takes 30+ seconds per utterance on CPU. The fastest model (Whisper-tiny, 39M params) takes 0.3s but has ~8% WER. Edge deployment demands both low latency AND high accuracy — a fundamental tension.

**Challenge 4: Small Array Aperture**
A 14mm mic spacing (phone) has a spatial aliasing frequency of ~12kHz. Above this, the array cannot distinguish direction. For a 4-mic linear array with 15mm spacing, the total aperture is 45mm — the array gain is only ~6dB (theoretical maximum is 10*log10(N) = 6dB for 4 mics). This limits how much spatial filtering can improve SNR.

### 1.3 The constraints we operate under

| Constraint | Value | Source |
|------------|-------|--------|
| Max parameters | <50M | Edge hardware (Raspberry Pi 4, Jetson Nano, phone NPU) |
| Max latency (end-to-end) | <500ms | Real-time conversational requirement |
| Always-on power | <5mW | Battery-powered always-on detection |
| Active pipeline power | <5W | During active processing |
| Mic count | 2-4 | Phone/smart speaker form factor |
| Mic spacing | 14-58mm | Real edge device dimensions |
| Sample rate | 16kHz | Standard for speech (8kHz-16kHz sufficient for human voice 80-8000Hz) |
| Target WER | <15% | Usable transcription quality |
| Target SSL error | <10 degrees | Meaningful spatial filtering |
| Target DER | <15% | Accurate speaker attribution |

---

## 2. THE PIPELINE — WHY THIS ARCHITECTURE

### 2.1 Why a modular cascade, not end-to-end

**The argument for end-to-end:** A single neural network trained on (multichannel audio -> text) could learn all intermediate representations. This works well in unconstrained settings (cloud, unlimited compute).

**Why we use a modular cascade instead:**

1. **Interpretability requirement (EE678).** We must explain what happens at each stage using wavelet analysis. An end-to-end model is a black box.

2. **Edge hardware is heterogeneous.** Real edge devices have DSP + NPU + CPU. Each module can run on the best hardware unit:
   - VAD: DSP (always-on, micro-watts)
   - SSL + beamforming: DSP or NPU (low-latency signal processing)
   - Enhancement: NPU (lightweight neural or signal processing)
   - ASR: CPU/NPU (most compute-intensive)

3. **Failure isolation.** If SSL fails (wrong direction), a modular system can detect this (low confidence) and fall back to processing without beamforming. An end-to-end system would silently degrade.

4. **Ablation studies.** To write a publishable paper, we must show the contribution of each module. This requires swapping modules independently — impossible with end-to-end.

5. **Published evidence.** "On the Underestimated Potential of Modular Speech-to-..." (2025) shows systematically optimized cascaded pipelines achieve comparable or better WER than end-to-end, with sub-second latency.

### 2.2 The stages and what each one does

```
Stage 0: Always-On Listener (Wavelet VAD)
    Question: "Is anyone speaking?"
    Input: Raw mic signal
    Output: Binary speech/non-speech decision per frame
    Why needed: Saves power. Most of the time, nobody is speaking.
    Method: DWT sub-band energy ratio E(cA_3)/E(cD_1) > threshold
    Compute: ~20 multiply-accumulates per sample, <5mW

Stage 1: Sound Source Localization (SSL)
    Question: "Where are the speakers?"
    Input: Multichannel audio [n_mics, n_samples]
    Output: Direction of arrival (azimuth in degrees) per source
    Why needed: Tells the beamformer where to focus.
    Physics: Sound arrives at each mic with a different delay proportional
             to the angle of arrival. TDOA = d*sin(theta)/c where d is mic
             spacing and c = 343 m/s.
    Challenge: Reverberation creates false peaks in the cross-correlation.
    Current best: SRP-PHAT (classical), Cross3D-Edge CNN (neural)

Stage 2: Beamforming (Spatial Filtering)
    Question: "Can we focus on the target speaker's direction?"
    Input: Multichannel audio + estimated DOA from SSL
    Output: Single-channel audio with improved SNR toward DOA
    Why needed: Suppresses noise and interference from other directions.
    Physics: Delay-and-sum aligns signals from target direction, causing
             constructive interference. Signals from other directions
             destructively interfere. Array gain = 10*log10(N_mics) dB.
    Challenge: If SSL DOA is wrong by >15 degrees, the beamformer
              suppresses the target instead of the interference.
    Current best: MVDR (optimal for known noise statistics)

Stage 3: Speech Enhancement (Noise Reduction)
    Question: "Can we remove the remaining noise?"
    Input: Beamformed single-channel audio
    Output: Cleaned single-channel audio
    Why needed: Beamforming reduces directional noise but not diffuse noise.
    CRITICAL FINDING (V1): Enhancement can HURT in multi-speaker scenarios.
        Without enhancement: WER = 25%
        With wavelet enhancement: WER = 39%
        With spectral subtraction: WER = 110%
    Why it hurts: The enhancer treats the interfering speaker as noise and
                  removes frequency bands shared with the target speaker.
    Implication: Enhancement should be CONDITIONAL — only applied when
                 a single speaker is detected (via VAD/SSL).

Stage 4: Speaker Separation (for multi-speaker scenarios)
    Question: "Can we isolate each speaker's voice?"
    Input: Enhanced audio (or beamformed if enhancement is skipped)
    Output: Separated audio streams, one per speaker
    Why needed: ASR needs single-speaker audio. Multi-speaker degrades WER.
    Challenge: The "permutation problem" — which output corresponds to
              which speaker? Solved with PIT (permutation invariant training).
    Not yet implemented (V2).

Stage 5: Automatic Speech Recognition (ASR)
    Question: "What did they say?"
    Input: Clean single-speaker audio
    Output: Text transcription with timestamps
    Why needed: This is the end goal — converting speech to text.
    Challenge: Latency. Whisper-small takes 3000ms for 15s audio.
              Whisper-tiny takes 300ms but with higher WER.
    Current best: Whisper-tiny for edge (39M params, 0.3s, ~8% WER)

Stage 6: Speaker Diarization
    Question: "Who spoke when?"
    Input: Audio + ASR timestamps
    Output: Speaker-attributed transcript segments
    Why needed: Distinguishes "Speaker A said X" from "Speaker B said Y."
    Not yet implemented (V2).
```

### 2.3 Why this order matters

The pipeline order is NOT arbitrary. Each stage depends on the output of the previous:

```
SSL accuracy → Beamformer quality → Enhancement quality → ASR accuracy
```

**If SSL is wrong (>15 deg):** Beamformer points wrong direction, suppresses target. Everything downstream fails. This is why SSL is Priority 1 for V2.

**If beamformer fails:** Enhancement receives noisy multi-speaker audio, treats interference as noise, creates artifacts. ASR gets distorted input. This is exactly what happened in V1.

**The dependency chain means: fix from left to right.** Don't optimize ASR until the upstream modules provide clean audio.

---

## 3. THE METRICS — WHAT TO MEASURE AND WHY

### 3.1 First principles: what makes a metric useful?

A good metric must be:
1. **Correlated with the end goal.** If PESQ improves but WER gets worse, PESQ is misleading.
2. **Sensitive to the changes we make.** If swapping SSL algorithm doesn't change the metric, it's too coarse.
3. **Comparable across papers.** Use standard metrics so our results are reproducible.
4. **Computationally cheap.** We compute metrics after every run in the dashboard.

### 3.2 Module-level metrics and what they tell you

**SSL: Root Mean Square Angular Error (RMSAE)**
```
RMSAE = sqrt( (1/N) * sum_i (angular_error_i)^2 )
```
- Unit: degrees
- Range: 0 (perfect) to 180 (opposite direction)
- Why this and not MAE: RMSAE penalizes large errors more heavily. A system with one 90-degree error is worse than one with three 30-degree errors, and RMSAE reflects this.
- Target: <10 degrees (adequate for MVDR beamforming)
- Current V1: 48-80 degrees (POOR)

**Beamforming: SI-SDR improvement (SI-SDRi)**
```
SI-SDRi = SI-SDR(beamformed, reference) - SI-SDR(input_mic0, reference)
```
- Unit: dB
- SI-SDRi > 0 means the beamformer helped
- SI-SDRi < 0 means it made things worse (usually due to bad DOA)
- Target: >5 dB improvement
- Current V1: -15 to +3 dB (highly variable, depends on SSL accuracy)

**Enhancement: PESQ + STOI**
- **PESQ** (Perceptual Evaluation of Speech Quality): Models human auditory perception. Range -0.5 to 4.5. Correlates with MOS (Mean Opinion Score).
  - <2.0 = poor, 2.0-2.5 = acceptable, 2.5-3.5 = good, >3.5 = excellent
- **STOI** (Short-Time Objective Intelligibility): Predicts word intelligibility. Range 0-1.
  - <0.7 = poor, 0.7-0.8 = acceptable, 0.8-0.9 = good, >0.9 = excellent
- **CRITICAL:** PESQ and STOI can improve while WER gets worse. This happens when the enhancer makes the audio sound "better" to humans but introduces artifacts that confuse the ASR model. **Always check WER alongside PESQ/STOI.**

**ASR: Word Error Rate (WER) + Character Error Rate (CER)**
```
WER = (Substitutions + Deletions + Insertions) / N_reference_words
CER = same formula but at character level
```
- WER is the PRIMARY end-to-end metric. Everything else is secondary.
- CER is useful because CER < WER means the model gets characters right but makes word-boundary errors (fixable with language model).
- **Decomposition matters:** High deletions = ASR doesn't hear the speech. High substitutions = ASR confuses words. High insertions = ASR hallucinates words.
- Target: <15% WER
- Current V1: 21-110% (single speaker OK, multi-speaker terrible)

**System: Real-Time Factor (RTF) + Latency**
```
RTF = processing_time / audio_duration
```
- RTF < 1.0 means real-time capable
- RTF < 0.5 means room for additional processing
- Current V1: RTF = 0.20 (Whisper-small), 0.03 (Whisper-tiny)

**Compute: MACs (Multiply-Accumulate Operations)**
```
1 MAC = 1 multiplication + 1 addition = 2 FLOPs
```
- The standard measure of algorithmic compute cost for edge deployment
- Device-independent — counts operations regardless of hardware
- Used by Cross3D-Edge (127.1 MMACs), Conv-TasNet, and all edge papers
- Our pipeline: SSL ~0.5-2 MMACs, BF ~0.1 MMACs, Enhancement ~1-5 MMACs, ASR ~500-5000 MMACs
- **ASR dominates compute by 100-1000x** over signal processing modules

**Compute: Energy per Inference (estimated)**
```
Energy (mJ) = Device_TDP (W) x Latency (s) x 1000
```
- Upper-bound estimate using Thermal Design Power
- Real energy is lower (not all circuits active during inference)
- For precise measurement: use Joulescope (2 MHz) or Monsoon power monitor
- Reference TDP values:
  - Raspberry Pi 4B: 5W active, 1W idle
  - Phone NPU: 0.5W active
  - DSP always-on: 5mW
- Our pipeline (RPi4): ~15-25 mJ per inference (signal processing) + ~15,000 mJ for Whisper-small

**Compute: Parameters**
- Total trainable weight count across all neural modules
- Determines model storage size and loading time
- Constraint: <50M total for edge deployment
- V1: Whisper-tiny 39M, Whisper-small 244M (exceeds constraint!)

**Compute: Peak Memory**
- Maximum RAM usage during inference (via tracemalloc)
- Determines minimum device RAM requirement
- Edge devices: 512MB-2GB RAM typical

**Compute: Memory Bandwidth (MB/s)**
```
Memory_BW = Peak_Memory / Latency
```
- Shows if the computation is memory-bound or compute-bound
- Important for choosing between CPU, GPU, NPU execution

### 3.3 The one metric that matters most: End-to-End WER

All intermediate metrics (RMSAE, PESQ, STOI, SI-SDR) are proxies. The only thing that truly matters is: **can the system correctly transcribe what the target speaker said?**

When intermediate metrics improve but WER doesn't, the improvement is meaningless for the application. When intermediate metrics get worse but WER improves (e.g., skipping enhancement), the "worse" module should be preferred.

**The experimental protocol:** Every change is measured by its impact on WER. PESQ/STOI/RMSAE are diagnostic — they help explain WHY WER changed, not whether the change was good.

---

## 4. THE RESEARCH METHOD — HOW TO SYSTEMATICALLY IMPROVE

### 4.1 The Scientific Method Applied to Pipeline Engineering

```
Step 1: OBSERVE — Run the current pipeline. Record all metrics.
Step 2: HYPOTHESIZE — "Module X is the bottleneck because metric Y is poor."
Step 3: PREDICT — "Replacing X with X' should improve Y by Z, which should
                   reduce WER by W."
Step 4: EXPERIMENT — Change ONLY module X. Keep everything else constant.
Step 5: MEASURE — Record all metrics. Compare to prediction.
Step 6: CONCLUDE — Did WER improve? If yes, keep the change. If no, revert.
Step 7: LOG — Write everything in RESEARCH_LOG.md.
```

**The critical rule: CHANGE ONE THING AT A TIME.**

If you change SSL and enhancement simultaneously and WER improves, you don't know which change helped. Maybe both helped. Maybe one helped and the other hurt, but the net was positive. You'll make wrong decisions about future improvements.

### 4.2 How to identify the current bottleneck

Run the pipeline with **oracle (perfect) versions** of each module:

```
Experiment A: Use GROUND TRUTH DOA instead of SSL output.
    → If WER improves significantly: SSL is the bottleneck.
    → If WER doesn't change: SSL is not the bottleneck.

Experiment B: Use CLEAN SOURCE instead of beamformed audio.
    → If WER improves: Beamformer/enhancement is the bottleneck.

Experiment C: Use CLEAN SOURCE directly into ASR (skip all processing).
    → This gives the ASR ceiling — the best WER possible with perfect upstream.
    → If this WER is still bad: ASR model itself is the limit.
```

**V1 oracle analysis (from existing results):**
- Clean room, single speaker, Whisper-small: WER = 21.4%
- This means even with near-perfect conditions, Whisper-small has ~21% WER on this utterance. This is the ASR floor.
- Multi-speaker office: WER = 25% (without enhancement) — only 3.6% worse than clean room. This suggests the ROOM is not the main problem for this utterance; the interfering speaker is.

### 4.3 How to choose between algorithm variants

When comparing Algorithm A vs Algorithm B for a module, use:

**The Decision Matrix:**

| Criterion | Weight | How to measure |
|-----------|--------|----------------|
| WER impact | 40% | End-to-end WER change |
| Latency | 25% | Wall-clock time on target hardware |
| Parameters | 15% | Model size (must fit in <50M total) |
| Robustness | 10% | Performance variance across SNR x RT60 grid |
| Interpretability | 10% | Can we explain why it works using wavelets? |

**Example decision: GCC-PHAT vs SRP-PHAT vs MUSIC for SSL**

| Criterion | GCC-PHAT | SRP-PHAT | MUSIC |
|-----------|----------|----------|-------|
| WER impact | 80.8 deg -> high WER | Unknown | 48 deg -> medium WER |
| Latency | 9ms (fast) | 37ms (moderate) | 30-37ms (moderate) |
| Parameters | 0 | 0 | 0 |
| Robustness | Poor in reverb | Better in reverb | Needs source count |
| Interpretability | Simple phase analysis | Spatial spectrum | Eigendecomposition |
| **Score** | Low | TBD | Medium |

**The next experiment should be:** Run SRP-PHAT on the same office scene. If RMSAE < 30 deg, it wins over GCC-PHAT. Then test on the full SNR x RT60 grid.

### 4.4 When to stop optimizing a module

A module is "good enough" when:
1. Its module-level metric is in the "Good" range (DATASETS_AND_METRICS.md Section 4)
2. Improving it further doesn't improve WER by more than 1%
3. You have tested at least 3 variants (to avoid local minima)

**Example:** If SSL RMSAE is 8 degrees and switching from SRP-PHAT to Cross3D-Edge reduces it to 6 degrees, but WER changes from 12.3% to 12.1%, the improvement is not worth the added complexity (140K extra parameters, training required).

---

## 5. THE WAVELET CONTRIBUTION — WHY MULTI-RESOLUTION MATTERS

### 5.1 First principles: why wavelets for audio?

Speech is a **non-stationary signal** — its frequency content changes rapidly over time. A sustained vowel has energy concentrated below 3kHz. A consonant like "s" has energy above 4kHz. Speech activity detection, enhancement, and feature extraction all benefit from analyzing different frequency bands independently.

**STFT vs DWT:**
- STFT uses fixed-size windows (e.g., 32ms). Good frequency resolution at low frequencies requires long windows. Good time resolution requires short windows. You can't have both — this is the Heisenberg uncertainty principle for time-frequency analysis.
- DWT naturally provides **good time resolution at high frequencies** (where transients like consonants live) and **good frequency resolution at low frequencies** (where vowel formants live). This matches the structure of speech.

**Practical advantage for edge:** DWT is O(N) per sample. STFT is O(N log N) per frame. For always-on processing where every micro-watt counts, DWT is preferable.

### 5.2 Where wavelets appear in our pipeline

**VAD (Stage 0):** The energy ratio E(cA_J)/E(cD_1) exploits the fact that speech concentrates energy in low-frequency approximation coefficients while broadband noise distributes energy uniformly. This ratio is a cheap, interpretable VAD feature.

**Enhancement (Stage 3):** DWT soft thresholding (Eq. 3.2-3.5 in PIPELINE_ALGORITHM.md) applies level-dependent thresholds to detail coefficients. Noise, being broadband, appears in all levels. Speech, being structured, concentrates in specific levels. Thresholding removes noise while preserving speech structure.

**Interpretability (all stages):** At each pipeline stage, we compute the wavelet scalogram — a time-scale representation showing how energy distributes across DWT sub-bands. This shows:
- What information the beamformer preserves vs discards
- Which frequency bands the enhancer modifies
- Whether the ASR input still contains the speech formant structure

### 5.3 The wavelet ablation experiment

To demonstrate that wavelets provide measurable improvement:

```
Experiment: Enhancement with DWT vs without DWT
    Condition A: STFT-domain spectral subtraction (baseline)
    Condition B: DWT sub-band thresholding (our method)
    Condition C: No enhancement (skip)

    Measure: PESQ, STOI, WER, latency, FLOPs per frame
    Scene: Single speaker, SNR=15dB, RT60=0.4s

    Expected result:
    - STFT: PESQ ~2.0, WER ~18%
    - DWT: PESQ ~2.1 (marginal improvement), WER ~17%
    - None: PESQ ~1.5 (no enhancement), WER ~15-20%

    Key insight: DWT and STFT give similar quality, but DWT uses
    30% fewer FLOPs — meaningful for edge deployment.
```

---

## 6. THE EXPERIMENTAL ROADMAP — WHAT TO DO NEXT, IN ORDER

### Phase 1: Establish the ASR Ceiling (1 day)

**Goal:** Know the best possible WER before optimizing upstream.

```
Experiment 1.1: Clean LibriSpeech audio -> Whisper-tiny -> WER
Experiment 1.2: Clean LibriSpeech audio -> Whisper-base -> WER
Experiment 1.3: Clean LibriSpeech audio -> Whisper-small -> WER
```

This tells us: "Even with perfect upstream, WER cannot be better than X%."

### Phase 2: SSL Comparison on Grid (3 days)

**Goal:** Find the best classical SSL method.

```
Experiment 2.1: GCC-PHAT on 18-point SNR x RT60 grid
Experiment 2.2: SRP-PHAT on same grid
Experiment 2.3: MUSIC on same grid (with n_sources=1)
```

Measure: RMSAE per condition. Find which method has lowest average and lowest worst-case error. The winner becomes the baseline for further improvement.

### Phase 3: Beamformer Comparison (2 days)

**Goal:** Compare DS vs MVDR, conditioned on SSL quality.

```
Experiment 3.1: DS beamformer with ground-truth DOA -> WER
Experiment 3.2: MVDR beamformer with ground-truth DOA -> WER
Experiment 3.3: DS beamformer with SSL-estimated DOA -> WER
Experiment 3.4: MVDR beamformer with SSL-estimated DOA -> WER
```

If 3.1 and 3.2 show big improvement over 3.3/3.4, then SSL accuracy is the bottleneck (not the beamformer).

### Phase 4: Enhancement Ablation (2 days)

**Goal:** Determine if enhancement helps or hurts, and under what conditions.

```
Experiment 4.1: No enhancement -> WER
Experiment 4.2: Spectral subtraction (alpha=1.0, 1.5, 2.0, 3.0) -> WER
Experiment 4.3: Wavelet (bior2.2, levels=2,3,4, threshold=0.5,1.0,2.0) -> WER
Experiment 4.4: Enhancement only when single speaker (VAD-gated) -> WER
```

The key question is: **does enhancement help ASR?** If not, skip it.

### Phase 5: Latency Optimization (1 day)

**Goal:** Achieve <500ms total latency.

```
Experiment 5.1: Whisper-tiny on enhanced audio -> WER and latency
Experiment 5.2: Whisper-base on enhanced audio -> WER and latency
```

Choose the model that achieves WER < 15% with latency < 500ms.

### Phase 6: Multi-Speaker Separation (if needed, 2 weeks)

**Only start this if Phases 1-5 achieve good single-speaker results.**

```
Experiment 6.1: Conv-TasNet on LibriMix -> SI-SDRi
Experiment 6.2: Conv-TasNet + SSL spatial features -> SI-SDRi
Experiment 6.3: Full pipeline with separation -> WER
```

### Phase 7: Paper Experiments and Figures (1 week)

Generate all figures and tables for the paper:
- SNR x RT60 heatmaps per metric per module variant
- Wavelet scalograms at each pipeline stage
- Ablation tables showing contribution of each module
- Latency breakdown charts
- Accuracy vs. latency Pareto frontier

---

## 7. OPEN QUESTIONS — WHAT WE DON'T KNOW YET

These are research questions. The experiments above will answer them.

1. **Does MVDR outperform DS when SSL is accurate (<10 deg)?**
   We hypothesize yes, but haven't tested. If MVDR doesn't help, the mm-scale array is too small for optimal beamforming.

2. **Is wavelet enhancement better than STFT enhancement in terms of FLOPs/WER tradeoff?**
   We hypothesize wavelets are cheaper for similar quality. Need to measure on identical conditions.

3. **What is the SSL accuracy threshold below which beamforming hurts?**
   Our V1 data suggests ~15-20 degrees, but we need a systematic sweep.

4. **Can wavelet VAD replace neural VAD for always-on detection?**
   We hypothesize wavelet VAD achieves 88-93% F1 at 100x lower power. Need to benchmark.

5. **Does the feedback loop (SSL -> separation) actually improve WER in our system?**
   Literature says yes (+2-5 dB SI-SDRi), but with accurate SSL. In our system with inaccurate SSL, it might not help.

---

## 8. HOW TO READ THE OTHER DOCUMENTATION FILES

| When you want to... | Read... |
|---------------------|---------|
| Understand the math of a specific algorithm | `docs/PIPELINE_ALGORITHM.md` (equation numbers) |
| See what experiments have been run and their results | `docs/RESEARCH_LOG.md` |
| Know what datasets to use and how to compute metrics | `docs/DATASETS_AND_METRICS.md` |
| Find papers to cite or read | `docs/REFERENCES.md` |
| Understand the simulation testbench | `docs/ACOUSTIC_LAB.md` |
| See the overall project rules and architecture | `CLAUDE.md` |
| Understand what was built in V1 | `docs/V1_IMPLEMENTATION.md` |
| Run experiments interactively | `python -m edge_audio_intelligence.dashboard.app` |

---

*This document is the intellectual foundation of the project. Update it when understanding deepens. Every statement should be backed by either physics, math, or experimental evidence.*
