# PIPELINE_ALGORITHMS.md — Algorithm Reference for Every Pipeline Module
# Referenced by: CLAUDE.md Section 1

> **Purpose:** This file contains the complete mathematical specification, design rationale, and implementation notes for every algorithm in the pipeline. When implementing a module, read its section here first. When debugging, check the math here. Every equation is numbered for cross-referencing in code docstrings.

## IMPLEMENTATION STATUS (as of 2026-03-20)

| Algorithm | Status | File |
|-----------|--------|------|
| Signal Model (Eq. 0.1-0.3) | IMPLEMENTED | `algorithms/rir.py`, `algorithms/signal_mixing.py` |
| GCC-PHAT (Eq. 1.1-1.3) | IMPLEMENTED | `modules/ssl/gcc_phat.py` |
| SRP-PHAT (Eq. 1.4-1.8) | IMPLEMENTED | `modules/ssl/srp_phat.py` |
| MUSIC (Eq. 1.9-1.12) | IMPLEMENTED | `modules/ssl/music.py` |
| Cross3D-Edge CNN | PLANNED V2 | Not yet built |
| Delay-and-Sum (Eq. 2.1-2.2) | IMPLEMENTED | `modules/beamforming/delay_and_sum.py` |
| MVDR (Eq. 2.3-2.5) | IMPLEMENTED | `modules/beamforming/mvdr.py` |
| ADL-MVDR | PLANNED V2 | Not yet built |
| Spectral Subtraction (Eq. 3.1) | IMPLEMENTED | `modules/enhancement/spectral_subtraction.py` |
| DWT Sub-band Enhancement (Eq. 3.2-3.5) | IMPLEMENTED | `modules/enhancement/wavelet_enhancement.py` |
| DeepFilterNet | PLANNED V2 | Not yet built |
| Conv-TasNet (Eq. 4.1-4.4) | PLANNED V2 | Not yet built |
| Emformer (Eq. 5.1-5.3) | PLANNED V2 | Not yet built |
| Whisper (offline) | IMPLEMENTED | `modules/asr/whisper_offline.py` |
| pyannote Diarization (Eq. 6.1-6.2) | PLANNED V2 | Not yet built |
| DWT Math (Eq. 7.1-7.5) | IMPLEMENTED | `wavelet/dwt_features.py`, `wavelet/analysis.py` |

---

## 0. SIGNAL MODEL

The fundamental signal model governs all downstream algorithms.

**Eq. 0.1 — Received signal at microphone m:**
```
z_m(t) = Σ_{s=1}^{S} [h_{s,m}(t) * x_s(t)] + n_m(t)
```
Where:
- `x_s(t)` = clean source signal from speaker s (unknown)
- `h_{s,m}(t)` = room impulse response from source s to mic m (unknown)
- `n_m(t)` = additive noise at mic m
- `S` = number of sources
- `*` = convolution

**Eq. 0.2 — STFT domain representation:**
```
Z_m(t,f) = Σ_{s=1}^{S} H_{s,m}(f) · X_s(t,f) + N_m(t,f)
```
Under the narrowband approximation, convolution becomes multiplication per frequency bin. This holds when the STFT window is much longer than the maximum time delay across the array.

**Eq. 0.3 — Far-field steering vector:**
```
a(θ,φ,f) = [e^{-j2πf·τ_1(θ,φ)}, ..., e^{-j2πf·τ_M(θ,φ)}]^T
```
Where `τ_m(θ,φ) = (d_m · u(θ,φ)) / c` is the propagation delay to mic m, `d_m` is the mic position vector, `u(θ,φ) = [cos(φ)cos(θ), cos(φ)sin(θ), sin(φ)]^T` is the unit direction vector, and `c = 343 m/s`.

---

## 1. SOUND SOURCE LOCALIZATION (SSL)

### 1.1 GCC-PHAT

**Eq. 1.1 — GCC-PHAT between mic pair (m, m'):**
```
R_{m,m'}(τ) = IFFT{ X_m(f) · X*_{m'}(f) / |X_m(f) · X*_{m'}(f)| }
```
The PHAT weighting `1/|X_m·X*_{m'}|` normalizes the cross-spectrum to unit magnitude, keeping only phase information. This makes GCC-PHAT robust to spectral coloring but sensitive to spatial aliasing.

**Eq. 1.2 — TDoA estimation:**
```
Δτ_{m,m'} = argmax_τ R_{m,m'}(τ)
```

**Eq. 1.3 — DoA from TDoA (for linear array):**
```
θ = arcsin(Δτ · c / d_{m,m'})
```
where `d_{m,m'}` = distance between mics m and m'.

**Implementation notes:**
- Use zero-padded FFT for sub-sample TDoA resolution
- Parabolic interpolation around the GCC peak improves precision
- For M>2 microphones, compute GCC for all M(M-1)/2 pairs, then average or vote

### 1.2 SRP-PHAT

**Eq. 1.4 — SRP-PHAT power at candidate location q:**
```
P(q) = Σ_{(m,m'):m>m'} ∫ G_{m,m'}(ω) · e^{jω(τ^q_m - τ^q_{m'})} dω
```

**Eq. 1.5 — Time-domain SRP (TD-SRP):**
```
P(q) = 2 · Σ_{(m,m'):m>m'} G_{m,m'}(Δt_{m,m'}(q))
```
Where `G_{m,m'}` is the time-domain GCC-PHAT and `Δt_{m,m'}(q)` is the expected TDoA for location q.

**Eq. 1.6 — LC-SRP (Low-Complexity SRP) with sinc interpolation:**
```
G^{appr}_{m,m'}(τ) = Σ_{n∈N_{m,m'}} G_{m,m'}(nT) · sinc(τ/T - n)
```

**Eq. 1.7 — LC-SRP-Edge (from Yin & Verhelst, 2025, Eq. 8):**
```
G^{appr}_{m,m'}(τ) = Σ_{n=0}^{N_samp} [
    W^n_sinc · (τ/T) · Σ_k ℜ[G(k)] · ℜ[e^{j2πk/K · nT}]
  + W^n_sinc · n     · Σ_k ℑ[G(k)] · ℑ[e^{j2πk/K · nT}]
] · cos(nπ)

where W^n_sinc = 2·sin(πτ/T) / [π(τ/T - n)(τ/T + n)]
```
This halves computation and memory vs. LC-SRP by exploiting conjugate symmetry.

**Eq. 1.8 — DoA estimation from SRP map:**
```
(θ*, φ*) = argmax_{(θ,φ)} P(θ, φ)
```
For multi-source: find multiple local maxima above a threshold.

### 1.3 MUSIC

**Eq. 1.9 — Spatial covariance matrix:**
```
R(f) = (1/N) · Σ_{n=1}^{N} X(f,n) · X(f,n)^H
```
where `X(f,n) = [X_1(f,n), ..., X_M(f,n)]^T` is the multichannel STFT vector.

**Eq. 1.10 — Eigenvalue decomposition:**
```
R(f) = U_s · Λ_s · U_s^H + U_n · Λ_n · U_n^H
```
`U_s` = signal subspace eigenvectors (S largest eigenvalues)
`U_n` = noise subspace eigenvectors (M-S smallest eigenvalues)

**Eq. 1.11 — MUSIC pseudospectrum:**
```
P_MUSIC(θ,φ,f) = 1 / (a(θ,φ,f)^H · U_n · U_n^H · a(θ,φ,f))
```
Peaks occur where the steering vector is orthogonal to the noise subspace (i.e., lies in the signal subspace).

**Eq. 1.12 — Wideband MUSIC (average across frequency):**
```
P_MUSIC(θ,φ) = (1/K) · Σ_{f=1}^{K} P_MUSIC(θ,φ,f)
```

### 1.4 Cross3D-Edge Neural SSL

**Architecture (from Yin & Verhelst, 2025, Fig. 3b):**
```
Input: SRP map + SRP-max coordinates → [3, T, Res1, Res2]
  ↓
Input_Conv: 3D CNN, 32 filters 5×5×5, PReLU
  ↓ [32, T, Res1, Res2]
Cross_Conv: N parallel 3D CNN blocks, C filters 5×3×3, PReLU
  Branch A: MaxPool 1×1×2 (azimuth reduction)
  Branch B: MaxPool 1×2×1 (elevation reduction)  
  ↓ Flatten & Concatenate
Output_Conv: Depthwise 1D CNN, 4C filters 1×5, dilation=2, PReLU
  ↓ 1D CNN, 3 filters 1×5, dilation=2, Tanh
Output: DoA as 3D Cartesian coordinates → [3, T]
```

**Key design parameters (from Cross3D-Edge ablation):**
- Resolution: 8×16 (best accuracy-efficiency tradeoff)
- Channel size C=16 (EM variant): 140K params, 24.41 MFLOPs/frame
- Sampling rate: 16kHz (fs=8kHz degrades accuracy significantly)
- Training: Adam optimizer, lr=0.0005→0.0001 at epoch 40, batch 5→10
- Loss: MSE on normalized Cartesian coordinates

### 1.5 Physics-Informed Neural SSL

**Pipeline:** Multichannel → STFT → CPS matrix (Eq. 1.9) → EVD (Eq. 1.10) → Eigenvectors → Neural mapping → DoA

Instead of computing MUSIC spectrum directly, feed eigenvectors to MLP that learns direction-sensitive activations. Process each frequency bin independently (sub-band layers), then fuse across frequencies (integration layer).

---

## 2. BEAMFORMING

### 2.1 Delay-and-Sum

**Eq. 2.1 — Delay-and-sum output:**
```
y_DS(t) = (1/M) · Σ_{m=1}^{M} z_m(t - δ_m)
```
where `δ_m = (d_m · u(θ_target)) / c` is the steering delay.

**Eq. 2.2 — Frequency domain:**
```
Y_DS(f) = w_DS(f)^H · Z(f)
w_DS(f) = (1/M) · a(θ_target, f)
```

**Array gain:** `AG = M` (in terms of power, i.e., +3dB per doubling of mics).

### 2.2 MVDR (Minimum Variance Distortionless Response)

**Eq. 2.3 — MVDR beamformer weights:**
```
w_MVDR(f) = [Φ_nn(f)^{-1} · a(θ,f)] / [a(θ,f)^H · Φ_nn(f)^{-1} · a(θ,f)]
```
where `Φ_nn(f)` is the noise spatial covariance matrix.

**Eq. 2.4 — Noise covariance estimation:**
```
Φ_nn(f) = (1/T_n) · Σ_{t∈silent} Z(t,f) · Z(t,f)^H
```
Estimated from frames where VAD indicates no speech. Requires reliable VAD.

**Eq. 2.5 — MVDR output:**
```
Y_MVDR(t,f) = w_MVDR(f)^H · Z(t,f)
```

**Practical issues:**
- Φ_nn may be singular if estimated from too few frames → add diagonal loading: `Φ_nn + εI`
- Steering vector a(θ) uses the DoA from SSL → SSL errors directly degrade MVDR performance
- For time-varying noise, use exponential moving average: `Φ_nn(t) = α·Φ_nn(t-1) + (1-α)·Z(t)Z(t)^H`

### 2.3 ADL-MVDR (All Deep Learning MVDR)

Replace matrix inversion with two RNNs:
- RNN 1: Estimates frame-level beamforming filter from multichannel input
- RNN 2: Estimates relative transfer function (RTF) for steering
- Complex ratio filtering (CRF) for stable joint training

**Reference:** Zhang et al., "ADL-MVDR: All Deep Learning MVDR Beamformer," ICASSP 2021.

---

## 3. SPEECH ENHANCEMENT

### 3.1 Spectral Subtraction (Baseline)

**Eq. 3.1:**
```
|Ŝ(t,f)|² = max(|Y(t,f)|² - α·|N̂(f)|², β·|Y(t,f)|²)
```
where `|N̂(f)|²` is estimated from noise-only frames, `α` is oversubtraction factor (1.0-2.0), `β` is spectral floor (0.01-0.1).

### 3.2 DWT Sub-band Enhancement (Wavelet Contribution)

**Eq. 3.2 — J-level DWT decomposition:**
```
x(t) → {cA_J, cD_J, cD_{J-1}, ..., cD_1}
```
Using CDF 5/3 wavelet (biorthogonal, low-complexity, used in JPEG2000).

For J=3, fs=16kHz:
- cA_3: 0–1kHz (fundamental, voicing)
- cD_3: 1–2kHz (first formant)
- cD_2: 2–4kHz (consonants, second/third formants)
- cD_1: 4–8kHz (fricatives, noise-dominant)

**Eq. 3.3 — Noise estimation per sub-band (MAD estimator):**
```
σ_j = median(|cD_j|) / 0.6745
```
The highest-frequency band cD_1 is used as the primary noise reference (most noise-dominated).

**Eq. 3.4 — Level-dependent soft thresholding:**
```
cD̃_j(k) = sign(cD_j(k)) · max(|cD_j(k)| - λ_j, 0)
λ_j = σ_j · √(2 · log(N_j))   [VisuShrink universal threshold]
```
where `N_j` is the number of coefficients at level j.

**Eq. 3.5 — Reconstruction:**
```
x̃(t) = IDWT({cA_J, cD̃_J, cD̃_{J-1}, ..., cD̃_1})
```

**Advanced variant:** Replace fixed thresholding with a learned CNN that takes sub-band features and predicts per-coefficient masks. This is the WA-FSN approach (2025 paper).

### 3.3 DeepFilterNet

Operates in ERB (Equivalent Rectangular Bandwidth) domain:
- Group STFT bins into ~32 ERB bands
- GRU processes ERB-scale features → predicts real-valued gain per band
- Deep filtering: per-bin learned FIR filter (5 taps) for fine spectral correction
- ~2M parameters, real-time on CPU

---

## 4. SPEAKER SEPARATION

### 4.1 Conv-TasNet

**Eq. 4.1 — Encoder:**
```
W = ReLU(Conv1D(x, N=512, L=16, stride=8))  → [N, T']
```

**Eq. 4.2 — Separator (TCN with masks):**
```
M_i = Sigmoid(TCN(W))  → [N, T'] for each source i
```
TCN: B=128 bottleneck, H=512 hidden, P=3 kernel, X=8 blocks, R=3 repeats.

**Eq. 4.3 — Decoder:**
```
ŝ_i(t) = ConvTranspose1D(M_i ⊙ W, L=16, stride=8)
```

**Eq. 4.4 — SI-SDR loss with PIT:**
```
s_target = (<ŝ, s> / ||s||²) · s
e_noise = ŝ - s_target
SI-SDR = 10 · log10(||s_target||² / ||e_noise||²)

L = -max_{π∈Perms} Σ_i SI-SDR(ŝ_{π(i)}, s_i)
```

### 4.2 Spatial Conv-TasNet (SSL-Guided Separation)

**Eq. 4.5 — Interaural Phase Difference (IPD) feature:**
```
IPD_{m,ref}(t,f) = angle(Z_m(t,f) · Z*_ref(t,f))
```
where ref is the reference microphone (typically mic 0).

**Eq. 4.6 — Interaural Level Difference (ILD) feature:**
```
ILD_{m,ref}(t,f) = 20 · log10(|Z_m(t,f)| / |Z_ref(t,f)|)
```

**Integration:** Concatenate IPD and ILD features (derived from SSL DoA) with the encoder output W before feeding to the TCN separator. The DoA from SSL determines the expected IPD pattern for each source, providing strong spatial prior for separation.

---

## 5. STREAMING ASR

### 5.1 Emformer Architecture

**Eq. 5.1 — Segment-based attention:**
```
For segment i with center frames C_i, memory M, and right context R:
  Context = [M_{i-1}, C_i, R_i]
  Attention(Q, K, V) = softmax(Q·K^T/√d) · V
  where Q comes from C_i, K and V come from full Context
```

**Eq. 5.2 — Memory update:**
```
M_i = Summary(C_i)  [learned linear projection of segment output]
Memory bank: [M_{i-L}, ..., M_{i-1}]  (L = memory length)
```

**Eq. 5.3 — RNN-T decoder:**
```
P(y_u | y_{1:u-1}, x_{1:t}) = Joint(Encoder(x_t), Predictor(y_{u-1}))
Joint(h_enc, h_pred) = Linear(tanh(Linear(h_enc) + Linear(h_pred)))
```

### 5.2 CTC Loss

**Eq. 5.4:**
```
L_CTC = -log P(y | x) = -log Σ_{π∈B^{-1}(y)} Π_t P(π_t | x_t)
```
where `B^{-1}(y)` is the set of all frame-level alignments that collapse to label sequence y (after removing blanks and repeated characters).

---

## 6. SPEAKER DIARIZATION

### 6.1 pyannote Pipeline

**Stage 1 — Segmentation:** PyanNet processes 5s chunks → frame-level speaker activity probabilities for up to 3 speakers per chunk.

**Stage 2 — Embedding:** ECAPA-TDNN extracts 192-dim speaker embeddings per speech segment.

**Stage 3 — Clustering:** Agglomerative hierarchical clustering with centroid linkage. Number of clusters determined by stopping threshold.

**Eq. 6.1 — Cosine similarity for clustering:**
```
sim(e_i, e_j) = (e_i · e_j) / (||e_i|| · ||e_j||)
```

**Eq. 6.2 — DER computation:**
```
DER = (t_miss + t_fa + t_confusion) / t_total
```
- `t_miss`: speech that was not detected
- `t_fa`: non-speech detected as speech
- `t_confusion`: speech attributed to wrong speaker

### 6.2 Spatial-Assisted Diarization

Concatenate SSL DoA features with speaker embeddings before clustering. For each speech segment, append the mean DoA during that segment as additional feature dimensions:

```
e_spatial = [e_speaker (192-dim), mean_azimuth, mean_elevation]  → 194-dim
```

This helps separate speakers with similar voices but different locations.

---

## 7. WAVELET MATHEMATICS REFERENCE

### 7.1 Discrete Wavelet Transform

**Eq. 7.1 — DWT decomposition (analysis):**
```
cA_j[k] = Σ_n x[n] · g[n - 2k]    (low-pass / approximation)
cD_j[k] = Σ_n x[n] · h[n - 2k]    (high-pass / detail)
```
where g = low-pass filter, h = high-pass filter, and the signal is downsampled by 2 at each level.

**Eq. 7.2 — IDWT reconstruction (synthesis):**
```
x[n] = Σ_k cA_j[k] · g̃[n - 2k] + Σ_k cD_j[k] · h̃[n - 2k]
```
where g̃, h̃ are the synthesis filters.

**Eq. 7.3 — CDF 5/3 filter coefficients:**
```
Analysis low-pass:  [-1/8, 1/4, 3/4, 1/4, -1/8]
Analysis high-pass: [-1/2, 1, -1/2]
```
These are symmetric, have compact support, and require only additions and shifts — ideal for low-power implementation.

### 7.2 Sub-band Energy

**Eq. 7.4 — Energy per sub-band per frame:**
```
E_j[t] = Σ_{k∈frame_t} |cD_j[k]|²
```

**Eq. 7.5 — Wavelet energy ratio (for VAD):**
```
R_VAD[t] = E(cA_J)[t] / E(cD_1)[t]
```
Speech: R_VAD >> 1 (energy concentrated in low frequencies).
Noise: R_VAD ≈ 1 (energy distributed uniformly).
Threshold: R_VAD > τ → speech active (τ determined empirically, typically 3-5).

---

## 8. COMPUTATIONAL COMPLEXITY REFERENCE

From Cross3D-Edge paper (Yin & Verhelst, 2025, Table 2):

| Algorithm | Complexity (specific part) |
|-----------|--------------------------|
| TD-SRP | N(N-1)/2 · (2K·log₂K + Q) |
| LC-SRP | N_samp · (2K + 4 + 2Q) |
| LC-SRP-Edge | (N_samp - N(N-1)/4) · (K + 2 + 2Q) |

Where N=mics, K=FFT points, Q=SRP candidates, N_samp=total interpolation samples.

For DWT: O(N) per level, O(J·N) total for J levels. Much cheaper than STFT which is O(N·log N).

---

*Last updated: [DATE]*  
*Referenced by: CLAUDE.md Section 1*  
*Every equation number is used in code docstrings: `# Implements Eq. 1.4 from PIPELINE_ALGORITHMS.md`*