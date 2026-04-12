# REFERENCES.md — Curated Bibliography with Implementation Notes
# Referenced by: CLAUDE.md Section 1

> **Purpose:** Every paper relevant to this project, organized by module, with notes on what's actually useful for implementation. This is NOT a dump of abstracts — it's a guide for what to read, what to implement, and what to cite.

---

## PRIORITY SYSTEM

- 🔴 **MUST READ + IMPLEMENT:** Core papers that define our approach. Code must be written based on these.
- 🟡 **MUST READ:** Important context and baselines. Cite in paper but don't necessarily implement.
- 🟢 **GOOD TO READ:** Useful background. Read if time permits.

## IMPLEMENTATION STATUS (Updated 2026-03-20)

**Used in code:** GCC-PHAT (Knapp & Carter), SRP-PHAT (DiBiase), MUSIC (Schmidt), Delay-and-Sum, MVDR, Spectral Subtraction (Boll), DWT Enhancement (CDF 5/3), Whisper (OpenAI), pyroomacoustics ISM.

**Not yet used:** Cross3D-Edge (Yin & Verhelst), Conv-TasNet (Luo & Mesgarani), DeepFilterNet, Emformer (Shi et al.), pyannote (Bredin), ADL-MVDR (Zhang et al.).

---

## 1. SOUND SOURCE LOCALIZATION

### 🔴 Yin & Verhelst (2025) — Cross3D-Edge
**"CNN-based Robust Sound Source Localization with SRP-PHAT for the Extreme Edge"**
arXiv:2503.02046

**Why critical:** This is our primary SSL implementation target. Provides complete architecture, training procedure, ablation methodology, and edge-deployment metrics on Raspberry Pi 4B. The LC-SRP-Edge algorithm halves SRP computation. Cross3D-Edge-Medium (C=16) achieves 140K params, 8.59ms/frame, ~6° RMSAE.

**What to implement:** LC-SRP-Edge (Eq. 8), Cross3D-Edge network (Fig. 3b), SNR×RT60 evaluation grid.

**Codebase:** https://github.com/DavidDiazGuerra/Cross3D

### 🔴 Grumiaux et al. (2022) — SSL Survey
**"A Survey of Sound Source Localization with Deep Learning Methods"**
J. Acoust. Soc. Am., 152(1), 107. arXiv:2109.03465

**Why critical:** Comprehensive survey of all SSL methods. Defines taxonomy (features, architectures, training). Our document 1 is based heavily on this.

**What to use:** Classification of input features, comparison tables, SELD paradigm description.

### 🔴 Grinstein et al. (2022) — Multi-Stream CNN for IoT
**"Real-Time Sound Source Localization for Low-Power IoT Devices Based on Multi-Stream CNN"**
IEEE Sensors Journal, 22(12), 12091-12102.

**Why critical:** Multi-stream block with parallel kernels (3,5,7) + aggregation gate. Depthwise separable convolutions for efficiency. Hybrid classification+regression output. Demonstrated 7.8ms/frame on Raspberry Pi.

**What to implement:** Multi-stream block architecture, efficiency enhancements. This is the architecture referenced in your Document 1.

### 🟡 Diaz-Guerra et al. (2020) — Cross3D Original
**"Robust Sound Source Tracking Using SRP-PHAT and 3D Convolutional Neural Networks"**
IEEE/ACM Trans. Audio Speech Lang. Process., 29, 300-311.

**Why important:** Baseline that Cross3D-Edge optimizes. Established SRP+CNN approach and gpuRIR simulator.

### 🟡 Adavanne et al. (2018) — SELDnet
**"Sound Event Localization and Detection of Overlapping Sources Using Convolutional Recurrent Neural Networks"**
IEEE JSTSP, 13(1), 34-48. arXiv:1807.00129

**Why important:** Defines the SELD task. CRNN architecture combining CNN feature extraction with GRU temporal modeling. Multi-task learning for joint detection and localization.

### 🟢 Dietzen et al. (2020) — LC-SRP
**"Low-Complexity Steered Response Power Mapping Based on Nyquist-Shannon Sampling"**
WASPAA 2021, 206-210.

**Why useful:** Original LC-SRP algorithm that Cross3D-Edge builds upon.

---

## 2. BEAMFORMING

### 🔴 Zhang et al. (2021) — ADL-MVDR
**"ADL-MVDR: All Deep Learning MVDR Beamformer for Target Speech Separation"**
ICASSP 2021, 6089-6093.

**Why critical:** Replaces classical MVDR matrix operations with neural networks. Two RNNs estimate beamforming weights and steering vectors. Complex ratio filtering for stable training.

**What to implement:** If time permits, replace classical MVDR with ADL-MVDR. Otherwise cite as future work.

### 🟡 Chang et al. (2024) — Deep Beamforming with ARROW Loss
**"Deep Beamforming for Speech Enhancement and Speaker Localization with ARROW Loss"**
Frontiers in Signal Processing, 4:1413983.

**Why important:** Novel array-response-aware loss function for training neural beamformers. Separable convolution architecture for efficiency.

### 🟡 Google (2021) — Sequential Neural Beamforming
**"Sequential Multi-Frame Neural Beamforming for Speech Separation and Enhancement"**
SLT 2021.

**Why important:** Alternating neural separation and spatial beamforming stages. Multi-frame context improves results significantly.

---

## 3. SPEECH ENHANCEMENT

### 🔴 Cherukuru & Mustafa (2024) — DWT-CNN-MCSE
**"CNN-based noise reduction for multi-channel speech enhancement system with DWT preprocessing"**
PeerJ Computer Science, 10:e1901.

**Why critical:** Direct precedent for our wavelet enhancement approach. DWT preprocessing + CNN for multichannel speech enhancement. Tested on AURORA and LibriSpeech at -10 to 20 dB SNR. Shows DWT+CNN outperforms beamforming+adaptive filtering baseline.

**What to implement:** DWT preprocessing pipeline, CNN denoising architecture.

### 🔴 WA-FSN (2025) — Wavelet-Enhanced Adaptive FullSubNet
**"Improving the Speech Enhancement Model with DWT Sub-Band Features in Adaptive FullSubNet"**
Electronics, 14(7), 1354.

**Why critical:** Replaces STFT complex spectrograms with DWT sub-band features. Shows +3.6% PESQ improvement with only 6 additional convolution ops. Low-frequency sub-bands contribute most to PESQ. THIS IS OUR WAVELET ENHANCEMENT REFERENCE.

**What to implement:** DWT feature extraction, sub-band selection strategy.

### 🟡 EMNLP 2024 — Speaking in Wavelet Domain
**"Speaking in Wavelet Domain: A Simple and Efficient Approach to Speed up Speech Enhancement"**
EMNLP 2024.

**Why important:** Uses wavelet domain for diffusion-based speech synthesis. Shows CDF 5/3 wavelet is effective for speech processing. Multi-level DWT decomposition into approximation and detail matrices.

### 🟢 Schröter et al. (2022) — DeepFilterNet
**"DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement"**
INTERSPEECH 2022.

**Why useful:** Our baseline enhancement model. ERB-scale processing + deep filtering.

---

## 4. SPEAKER SEPARATION

### 🔴 Luo & Mesgarani (2019) — Conv-TasNet
**"Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"**
IEEE/ACM Trans. ASLP, 27(8), 1256-1266.

**Why critical:** Our baseline separation model. Encoder-Separator-Decoder architecture. PIT training. Standard benchmark.

### 🟡 Taherian et al. (2024) — SSL-guided Separation
**"Leveraging Sound Localization to Improve Continuous Speaker Separation"**
ICASSP 2024.

**Why important:** KEY REFERENCE for our feedback loop. Shows how SSL DoA improves separation by providing spatial priors.

### 🟡 Quan & Li (2024) — SpatialNet
**"SpatialNet: Extensively Learning Spatial Information for Multichannel Joint Separation, Denoising and Dereverberation"**
IEEE/ACM Trans. ASLP, 32, 1310-1323.

### 🟡 Boeddeker et al. (2024) — TS-SEP
**"TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings"**
IEEE/ACM Trans. ASLP, 32, 1185-1197.

**Why important:** Joint diarization + separation. State-of-the-art on LibriCSS. Our reference for the feedback loop design.

---

## 5. STREAMING ASR

### 🔴 Shi et al. (2021) — Emformer
**"Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition"**
INTERSPEECH 2021, 2746-2750. arXiv:2010.10759

**Why critical:** Our primary ASR architecture. Segment-based attention with memory bank. Bounded memory, streaming capable.

### 🟡 Gulati et al. (2020) — Conformer
**"Conformer: Convolution-augmented Transformer for Speech Recognition"**
INTERSPEECH 2020.

**Why important:** State-of-the-art ASR encoder combining self-attention and convolution. Often used as encoder in RNN-T systems.

### 🟢 Shangguan et al. (2019) — Memory-Efficient Speech Recognition
**"Optimizing Speech Recognition for the Edge"**
arXiv (referenced in your paper list)

---

## 6. SPEAKER DIARIZATION

### 🔴 Bredin (2023) — pyannote.audio 2.1
**"pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe"**
INTERSPEECH 2023.

**Why critical:** Our diarization tool. Powerset loss for segmentation, ECAPA-TDNN embeddings, agglomerative clustering.

### 🟡 Kalda et al. (2024) — PixIT
**"PixIT: Joint Training of Speaker Diarization and Speech Separation from Real-world Multi-speaker Recordings"**
Odyssey 2024.

**Why important:** Joint diarization + separation training. Solves over-separation problem.

### 🟡 Cui (2024) — Joint Separation, Diarization, Recognition
**"Joint speech separation, diarization, and recognition for automatic meeting transcription"**
PhD Thesis, Université de Lorraine.

**Why important:** Most comprehensive treatment of our exact problem. MC-SA-ASR model. Phase information helps multichannel ASR. Essential reference for the full pipeline.

---

## 7. SYSTEM DESIGN AND EDGE DEPLOYMENT

### 🟡 Barker et al. (2018) — CHiME Challenge
**"The fifth 'CHiME' speech separation and recognition challenge"**
Computer Speech & Language, 53, 171-189.

### 🟡 Reddy et al. (2020) — DNS Challenge
**"The INTERSPEECH 2020 Deep Noise Suppression Challenge"**

### 🟡 Carletta et al. (2007) — AMI Meeting Corpus
**"The AMI Meeting Corpus"**
Language Resources and Evaluation.

---

## 8. TOOLS AND FRAMEWORKS

### Pyroomacoustics
Scheibler et al., "Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms," ICASSP 2018.

### gpuRIR
Diaz-Guerra et al., "gpuRIR: A Python library for room impulse response simulation with GPU acceleration," Multimedia Tools and Applications, 2021.

### ESPNet
Watanabe et al., "ESPnet: End-to-end speech processing toolkit," INTERSPEECH 2018.

---

## CITATION TEMPLATE

When citing in the paper, use IEEE format:
```
[N] A. Author, B. Author, "Paper title," Journal/Conference, vol. X, no. Y, pp. ZZ-ZZ, Year.
```

---

*Last updated: [DATE]*  
*Referenced by: CLAUDE.md Section 1*  
*Add new papers as they are discovered. Mark priority level immediately.*