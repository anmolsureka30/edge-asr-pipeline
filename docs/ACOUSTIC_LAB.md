# ACOUSTIC_LAB.md — Simulation Testbench Design & Evaluation Framework
# Referenced by: CLAUDE.md Section 1

> **Purpose:** This file defines exactly how the acoustic simulation lab works — room generation, signal synthesis, ground truth creation, probe-point evaluation, and visualization. The testbench is the foundation of all experiments. Every number in the paper comes from this system.

## CURRENT IMPLEMENTATION STATUS (Updated 2026-03-20)

**Architecture:** All mathematical algorithms are in `algorithms/` package (standalone, no pipeline dependencies). The `testbench/simulator.py` orchestrates these to build `AcousticScene` objects. The `dashboard/` package provides an interactive Dash web UI at `http://localhost:8050`.

**Microphone arrays (edge-realistic, mm-scale spacing):**
- Phone 2-mic: 14mm spacing (real smartphone dimensions)
- Edge device 4-mic linear: 15mm spacing (45mm total span)
- Smart speaker 4-mic circular: 29mm radius, 58mm diameter (ReSpeaker Lite form factor)

**Scene presets (in `dashboard/components/scene_setup.py`):**
- Clean Room: 4x3.5x2.8m, RT60=0.0 (anechoic), SNR=30dB, 1 speaker
- Office Meeting: 7x6x3m, RT60=0.4s, SNR=20dB, 2 speakers
- Noisy Cafe: 10x8x3.5m, RT60=0.6s, SNR=5dB, 2 speakers + noise
- Reverberant Hall: 12x9x4.5m, RT60=1.2s, SNR=15dB, 1 speaker

**Key file locations:**
- `algorithms/rir.py` — RIR generation (ISM), convolution, Sabine equation
- `algorithms/signal_mixing.py` — Multi-source mixing, SNR-calibrated noise
- `algorithms/doa.py` — DOA computation (azimuth, elevation, TDOA)
- `testbench/simulator.py` — Orchestrator (calls algorithms/ to build scenes)
- `testbench/evaluator.py` — Metrics at every probe point
- `dashboard/` — Interactive experiment management UI

---

## 1. DESIGN PHILOSOPHY

The testbench is NOT just a room simulator. It is a **controlled experimental apparatus** that:

1. Generates reproducible acoustic scenes with known ground truth at every level (DoA, clean sources, transcriptions, speaker labels).
2. Injects controlled degradations (noise, reverberation, speaker overlap) with precise parameters.
3. Evaluates pipeline quality at every inter-module boundary, not just end-to-end.
4. Enables A/B testing by changing one variable while holding everything else constant.
5. Scales from quick sanity checks (1 scene, 4 corner cases) to full evaluations (100+ scenes, 18 SNR×RT60 conditions).

---

## 2. SCENE GENERATION

### 2.1 Room Parameters

Each `AcousticScene` is fully characterized by:

```yaml
room:
  dimensions: [width, depth, height]  # meters, sampled from [3,3,2.5] to [10,8,6]
  wall_absorption: float              # 0.5 to 1.0 (higher = less reverberant)
  rt60: float                         # 0.0 to 1.5 seconds (derived from absorption)
  # Note: rt60 and absorption are coupled. Use Sabine's formula:
  # RT60 = 0.161 * V / (S * α_avg)
  # where V = volume, S = surface area, α_avg = average absorption

sources:
  n_sources: int                      # 1 to 4
  positions: list of [x, y, z]        # meters, >1m from microphone array
  trajectories: list of trajectory    # static or moving (sine oscillation)
  signals: list of str                # paths to dry audio files
  
noise:
  type: str                           # "diffuse", "point", "babble", "traffic"
  snr_db: float                       # 5 to 30 dB (measured at array center)
  noise_file: str                     # path to noise audio

microphone_array:
  type: str                           # "linear_2", "circular_4", "locata_12"
  position: [x, y, z]                 # meters
  geometry: np.ndarray [n_mics, 3]    # per-mic offsets from center
```

### 2.2 Microphone Array Geometries

Three standard configurations (matching real hardware scenarios):

**Linear 2-mic (phone-like):**
```python
# 2 mics, 7cm spacing (typical smartphone)
geometry = np.array([[-0.035, 0, 0], [0.035, 0, 0]])
```

**Circular 4-mic (smart speaker):**
```python
# 4 mics in a circle, radius 4.25cm (similar to ReSpeaker)
angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
radius = 0.0425
geometry = np.stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(4)], axis=1)
```

**LOCATA 12-mic Robot Head:**
```python
# 12 mics in 3D configuration, use LOCATA specification
# Positions from LOCATA dataset documentation (Evers et al., 2020)
# Max pair distance ~0.2m
```

### 2.3 Signal Synthesis Pipeline

The synthesis follows this exact procedure (matching Cross3D, Section 2.3):

```
FOR each scene:
  1. Load dry speech files from LibriSpeech (one per source)
  2. Generate room configuration (random or from preset)
  3. FOR each source s:
       FOR each microphone m:
         Compute RIR h_{s,m}(t) using Image Source Method
         Compute reverberant signal: y_{s,m}(t) = x_s(t) * h_{s,m}(t)
  4. Mix all sources at each microphone: 
       y_m(t) = Σ_s y_{s,m}(t)
  5. Generate noise n_m(t) at desired SNR:
       - For diffuse noise: use spatially uncorrelated noise at each mic
       - For point noise: generate RIR for noise source, convolve
       - Scale: n_m(t) *= (signal_power / noise_power) * 10^(-SNR/20)
  6. Final signal: z_m(t) = y_m(t) + n_m(t)
```

### 2.4 Ground Truth Generation

The scene generator automatically produces:

| Ground Truth | Format | Source | Used By |
|-------------|--------|--------|---------|
| DoA per source per frame | `[n_frames, n_sources, 2]` (azimuth, elevation) | Computed from source-array geometry | SSL evaluator |
| Clean speech per source | `[n_sources, n_samples]` | The dry audio before convolution | Enhancement/Separation evaluator |
| Reverberant clean per source per mic | `[n_sources, n_mics, n_samples]` | After RIR convolution, before mixing | Beamforming evaluator |
| Transcriptions per source | `list of str` | LibriSpeech annotations | ASR evaluator |
| Speaker activity timestamps | `list of (start, end, speaker_id)` | LibriSpeech VAD + source assignment | Diarization evaluator |
| Room parameters | `dict` | Scene config | Logging, reproducibility |

**Frame-level DoA computation:**
```python
# For each frame t (determined by STFT hop):
#   source_pos = trajectory(t)  # 3D position at time t
#   array_pos = mic_array_center  # 3D position
#   direction = source_pos - array_pos
#   azimuth = atan2(direction[1], direction[0])  # radians
#   elevation = atan2(direction[2], sqrt(direction[0]^2 + direction[1]^2))
#   doa[t] = [azimuth, elevation]
```

---

## 3. RIR GENERATION

### 3.1 Image Source Method (ISM)

Using pyroomacoustics for prototyping, gpuRIR for bulk generation.

```python
import pyroomacoustics as pra

room = pra.ShoeBox(
    room_dim,                    # [width, depth, height]
    fs=16000,                    # sample rate
    materials=pra.Material(absorption),  # wall absorption
    max_order=max_ism_order,     # reflection order (typically 10-20)
    ray_tracing=True,            # enable for late reverberation
    air_absorption=True          # realistic air absorption
)

room.add_source(source_position, signal=dry_audio)
room.add_microphone(mic_positions.T)  # [3, n_mics]
room.simulate()

rir = room.rir  # list of [n_mics] arrays, each is the RIR for that mic
recorded_signals = room.mic_array.signals  # [n_mics, n_samples]
```

### 3.2 RIR Validation

Before using any RIR, validate:
1. **RT60 check:** Measure RT60 from the generated RIR using Schroeder backward integration. Must match target within ±10%.
2. **Direct path check:** The first peak in the RIR must correspond to the direct path delay: t_direct = ||source_pos - mic_pos|| / 343.
3. **Energy check:** The RIR must have reasonable energy decay — no infinite reverb, no clipping.

### 3.3 Training Data Generation

For training neural modules (Cross3D-Edge, neural beamformer, etc.), generate large-scale datasets:

```
Training: 5000+ scenes
  - Random room: [3,3,2.5] to [10,8,6]
  - Random RT60: 0.0 to 1.5s continuous
  - Random SNR: 5 to 30 dB continuous
  - Random source positions (>1m from array)
  - Random trajectories (sine oscillation, matching Cross3D)
  - Audio: LibriSpeech train-clean-100

Validation: 500 scenes
  - Same distribution as training
  - Different audio files

Testing: Structured grid
  - 3 SNR levels × 6 RT60 levels = 18 conditions
  - 10 scenes per condition = 180 scenes total
  - Audio: LibriSpeech test-clean
  - Fixed random seeds for reproducibility
```

---

## 4. PROBE-POINT EVALUATION

### 4.1 Evaluation Architecture

The evaluator sits between every pair of modules and computes metrics:

```
Input Signal ──→ [SSL] ──probe──→ [Beamforming] ──probe──→ [Enhancement] ──probe──→ ...
                   │                    │                       │
                   ▼                    ▼                       ▼
              Angular Error       SIR Improvement          PESQ, STOI
              Detection Rate      PESQ vs clean            SI-SDR
```

### 4.2 Metric Implementations

All metrics are implemented in `utils/metrics.py`. Reference formulas:

**Angular Error:**
```
angular_error(θ_true, θ_est) = arccos(sin(φ_t)sin(φ_e) + cos(φ_t)cos(φ_e)cos(θ_t - θ_e))
where θ = azimuth, φ = elevation
RMSAE = sqrt(mean(angular_error^2))  over all frames
MAE = mean(angular_error) over all frames
```

**PESQ:** Use `pesq` Python package. Requires 16kHz signals. Range: -0.5 to 4.5.

**STOI:** Use `pystoi` package. Requires 10kHz signals (resample). Range: 0 to 1.

**SI-SDR:**
```
s_target = (<ŝ, s> / ||s||²) · s
e_noise = ŝ - s_target
SI-SDR = 10 · log10(||s_target||² / ||e_noise||²)
SI-SDRi = SI-SDR(enhanced) - SI-SDR(mixture)  # improvement
```

**WER:** Use `jiwer` package. WER = (S + D + I) / N, where S=substitutions, D=deletions, I=insertions, N=total reference words.

**DER:** Use `pyannote.metrics`. DER = (missed + false_alarm + confusion) / total_speech.

**Latency:** Use `time.perf_counter_ns()` with warm-up runs. Report median over 100 runs.

**FLOPs:** Use `thop` for PyTorch models, manual counting for signal processing algorithms.

### 4.3 Evaluation Reporting

Every experiment produces a standardized results structure:

```
results/
├── experiment_name/
│   ├── config.yaml              # Exact config used
│   ├── metrics.json             # All metric values
│   ├── per_scene_metrics.csv    # Metrics per scene
│   ├── summary_table.md         # Human-readable summary
│   └── figures/
│       ├── snr_rt60_heatmap.png
│       ├── module_comparison.png
│       └── latency_breakdown.png
```

---

## 5. VISUALIZATION DASHBOARD

The testbench includes a visualization module for debugging and paper figures:

### 5.1 Scene Visualization
- Room 3D view with source positions, mic array, DoA arrows (like your existing acoustic lab Fig. 3)
- RIR plots for each source-mic pair
- Spectrogram of mixed signal at each microphone

### 5.2 Pipeline Visualization
- Signal waveform and spectrogram at each probe point
- Wavelet scalogram at each probe point (course contribution)
- DoA estimation vs. ground truth over time
- Beampattern plot showing spatial selectivity

### 5.3 Results Visualization
- SNR×RT60 heatmaps for each metric
- Ablation bar charts comparing module variants
- Accuracy vs. latency scatter plots (Pareto front)
- Complexity×Error tradeoff metric (following Cross3D-Edge Table 3)

---

## 6. REAL-DATA TESTING

### 6.1 LOCATA Dataset (SSL + Beamforming)

- Tasks 1 and 3: static/moving single source with Robot Head array
- Room: 7.1×9.8×3.0m, RT60≈0.55s
- Metric: MAE on azimuth and elevation
- Comparison: Cross3D-Edge Table 6

### 6.2 AMI Meeting Corpus (Diarization)

- Real meeting recordings with overlap annotations
- 4-8 speakers per meeting
- Metric: DER
- Used for final system validation only

### 6.3 CHiME Data (Full Pipeline)

- If available, use CHiME-5/6/7 for realistic multi-speaker ASR evaluation
- Metric: WER with speaker attribution

---

## 7. COMMON PITFALLS AND GUARDS

1. **RIR length mismatch:** Ensure all RIRs are zero-padded to the same length before convolution. Different source-mic distances produce different RIR lengths.

2. **SNR computation:** Compute SNR AFTER room convolution (reverberant signal power), not on dry signals. Otherwise the actual SNR at the microphones will be wrong.

3. **Circular vs. linear arrays:** GCC-PHAT works well for azimuth-only with linear arrays but needs 3D arrays for elevation. Make sure you match the SSL algorithm to the array geometry.

4. **Frame alignment:** STFT frames for different modules may use different hop sizes. Establish a global frame rate (e.g., 10ms hop = 100 frames/sec) and resample all frame-level outputs to this rate before comparison.

5. **Source distance assumption:** SRP-PHAT and GCC-PHAT assume far-field (plane wave). For sources <1m from the array, the DoA estimate degrades. Enforce minimum distance in scene generation (Cross3D-Edge uses 1.0m minimum).

6. **Silence handling:** VAD must mark silent frames. SSL metrics should ONLY be computed on voiced frames (matching Cross3D-Edge evaluation). Report both "with silence" and "no-silence" metrics.

---

*Last updated: [DATE]*  
*Referenced by: CLAUDE.md Section 1*  
*Updates to this file require re-running affected experiments.*