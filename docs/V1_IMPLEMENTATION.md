# V1_IMPLEMENTATION.md — First Working Version Specification
# Referenced by: CLAUDE.md Section 1

> **Purpose:** This file defines exactly what V1 looks like — the minimum viable pipeline that produces end-to-end results. V1 is NOT fancy. V1 is correct, measurable, and complete. Every subsequent improvement builds on V1 as the baseline.

---

## 1. V1 DEFINITION

V1 is the simplest possible pipeline that:
- Takes multichannel audio as input
- Produces speaker-attributed transcription as output
- Measures every metric at every probe point
- Runs on simulated data with known ground truth
- Completes in reasonable time (<5 min per scene)

**V1 does NOT need:** neural beamforming, wavelet features, feedback loops, streaming ASR, real-time performance, edge optimization. Those come in V2+.

---

## 2. V1 MODULE SELECTION

| Stage | V1 Algorithm | Why This One | Future Upgrade |
|-------|-------------|--------------|----------------|
| SSL | GCC-PHAT | Already implemented in your acoustic lab, simplest, fast | SRP-PHAT → Cross3D-Edge |
| Beamforming | Delay-and-Sum | Simplest, built into pyroomacoustics | MVDR → Neural |
| Enhancement | Spectral Subtraction | No dependencies, fast, well-understood | DeepFilterNet → DWT-CNN |
| Separation | None (skip in V1) | Single-source V1 avoids permutation complexity | Conv-TasNet → Spatial Conv-TasNet |
| ASR | Whisper-small (offline) | Pre-trained, no training needed, good accuracy | Emformer streaming |
| Diarization | None (skip in V1) | Single-source V1 doesn't need it | pyannote.audio |

**V1 is a single-source pipeline.** Multi-source (separation + diarization) is V2.

---

## 3. V1 IMPLEMENTATION ORDER

### Step 1: Environment Setup (Day 1)

```bash
# Create conda environment
conda create -n edge_audio python=3.10
conda activate edge_audio

# Core dependencies
pip install numpy scipy matplotlib
pip install pyroomacoustics
pip install torch torchaudio
pip install openai-whisper    # or faster-whisper
pip install pesq pystoi       # metrics
pip install jiwer              # WER computation
pip install pyyaml             # config management
pip install librosa soundfile  # audio I/O

# Optional (for later versions)
# pip install pyannote.audio   # diarization (V2)
# pip install espnet            # neural beamforming (V3)
```

### Step 2: Testbench Core (Days 2-3)

Build in this order:
1. `testbench/scene.py` — AcousticScene dataclass that holds all scene parameters
2. `testbench/simulator.py` — Wrapper around pyroomacoustics that generates scenes
3. `testbench/mixer.py` — Signal mixing with noise at specified SNR
4. `utils/audio_io.py` — Load/save audio, resampling
5. `utils/metrics.py` — Implement angular_error, PESQ, STOI, SI-SDR, WER

**Validation test:** Generate one scene. Verify: (a) RIR has correct RT60, (b) mixed signal has correct SNR, (c) ground truth DoA matches source-array geometry.

### Step 3: SSL Module (Day 4)

Port your existing GCC-PHAT from the acoustic lab into the module framework:
1. `modules/ssl/base.py` — BaseSSL abstract class
2. `modules/ssl/gcc_phat.py` — GCC-PHAT implementation following Eq. 1.1-1.3

**Validation test:** Two-mic array, single source at known angle. Verify GCC-PHAT returns correct angle within 1°.

### Step 4: Beamforming Module (Day 5)

1. `modules/beamforming/base.py` — BaseBeamformer
2. `modules/beamforming/delay_and_sum.py` — DS beamformer using SSL output

**Validation test:** Source at 45°, noise at -45°. Verify: beamformed signal has higher SNR than any single microphone.

### Step 5: Enhancement Module (Day 6)

1. `modules/enhancement/base.py` — BaseEnhancer
2. `modules/enhancement/spectral_subtraction.py` — Basic spectral subtraction (Eq. 3.1)

**Validation test:** Known clean + noise at 10dB SNR. Verify PESQ of enhanced > PESQ of noisy.

### Step 6: ASR Module (Day 7)

1. `modules/asr/base.py` — BaseASR
2. `modules/asr/whisper_offline.py` — Wrapper around Whisper small

**Validation test:** Clean LibriSpeech utterance → Whisper → WER < 5%.

### Step 7: Pipeline Assembly (Day 8)

1. `pipeline/cascade.py` — CascadePipeline that chains modules
2. `pipeline/runner.py` — ExperimentRunner that generates scenes, runs pipeline, computes metrics

**The cascade pipeline:**
```python
class CascadePipeline:
    def __init__(self, ssl, beamformer, enhancer, asr):
        self.modules = [ssl, beamformer, enhancer, asr]
    
    def run(self, scene_data: dict) -> dict:
        data = scene_data
        for module in self.modules:
            data = module.process(data)
        return data
```

### Step 8: Baseline Experiment (Days 9-10)

Run the complete V1 pipeline on:
- 10 scenes per corner case (LL, HL, LH, HH) = 40 scenes
- Measure at every probe point
- Generate the **Baseline Results Table**

**This table is the most important output of V1. Everything else is measured against it.**

```
| Condition | SSL RMSAE | BF PESQ | Enh PESQ | ASR WER | Latency |
|-----------|-----------|---------|----------|---------|---------|
| LL (easy) |    ?.?°   |   ?.??  |   ?.??   |  ?.?%   |  ?ms    |
| HL (noisy)|    ?.?°   |   ?.??  |   ?.??   |  ?.?%   |  ?ms    |
| LH (reverb)|   ?.?°  |   ?.??  |   ?.??   |  ?.?%   |  ?ms    |
| HH (hard) |    ?.?°   |   ?.??  |   ?.??   |  ?.?%   |  ?ms    |
```

---

## 4. V1 → V2 TRANSITION

Once V1 baseline numbers are recorded, V2 adds:

1. **SRP-PHAT / Cross3D-Edge SSL** — Replace GCC-PHAT, re-measure full pipeline
2. **MVDR Beamforming** — Replace DS, re-measure
3. **DWT Enhancement** — Replace spectral subtraction, re-measure (wavelet contribution)
4. **Multi-source support** — Add Conv-TasNet separation, extend to 2-speaker scenes
5. **pyannote Diarization** — Add diarization module

Each upgrade is one experiment. Each experiment produces one row in the comparison table. After all V2 upgrades, you have the full ablation study.

---

## 5. V1 SUCCESS CRITERIA

V1 is "done" when:
- [x] Pipeline runs end-to-end on test scenes without errors (completed 2026-03-19)
- [x] All metrics (angular error, PESQ, STOI, SI-SDR, WER, CER) computed and logged (completed 2026-03-19)
- [x] Results saved in standardized JSON format in results/tables/ and results/run_history.json (completed 2026-03-19)
- [x] Baseline table filled in RESEARCH_LOG.md (completed 2026-03-20)
- [x] Bottleneck identified: SSL accuracy (40-80 deg in reverberant/multi-source) and enhancement quality (PESQ <1.2, actually hurts WER). ASR dominates latency (98% of total). See CLAUDE.md Section 9. (completed 2026-03-20)

**V1 COMPLETION DATE: 2026-03-20**

**V1 Actual Baseline Results (Clean Room, single speaker):**
- SSL: 3.9 deg angular error (Good)
- Enhancement PESQ: 1.92 (Acceptable)
- Enhancement STOI: 0.879 (Good)
- ASR WER: 21.4% (Acceptable)
- Total latency: 4566ms (Whisper-small dominates)

**V1 Identified Bottlenecks for V2:**
1. SSL degrades to 48-80 deg in multi-source/reverberant scenes
2. Enhancement (both Spectral Sub and Wavelet) makes WER WORSE in multi-speaker
3. Whisper-small takes ~3000ms; need Whisper-tiny or streaming ASR for edge

---

## 6. COMMON V1 ISSUES AND SOLUTIONS

**Issue: Whisper requires 16kHz mono audio.**
Solution: Always resample and downmix in the ASR module's `process()` method. Never modify upstream signals.

**Issue: GCC-PHAT fails with 2 mics for elevation.**
Solution: V1 uses azimuth-only DoA estimation with linear arrays. Elevation requires 3D arrays (LOCATA), which is V2.

**Issue: Spectral subtraction introduces musical noise.**
Solution: Use spectral floor β=0.1 and oversubtraction α=1.5. This is a known limitation — document it and show how DWT enhancement fixes it in V2.

**Issue: Pipeline is slow.**
Solution: V1 does NOT need to be real-time. Focus on correctness. Optimization is V3.

---

*Last updated: [DATE]*  
*Referenced by: CLAUDE.md Section 1*  
*After V1 is complete, this file becomes historical. V2 plans go in a new file.*