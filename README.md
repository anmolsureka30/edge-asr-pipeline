# Edge Audio Intelligence System

**A Multi-Resolution, Energy-Efficient Edge Audio Intelligence System for Continuous Low-Power Streaming Speech Recognition in Real-World Environments**

Course: EE678 Wavelets and Multiresolution Signal Processing, IIT Bombay  
Authors: Anmol Sureka (24B2470), Kushagra Bansal (24B2406) | Group 8

---

## 🏆 Final Pipeline Achievements (V2)

The system has completed rigorous structural upgrades over the baseline, implementing targeted physical metrics specifically to stabilize operations natively upon $<500$ms Edge computing frames maintaining robust Word Error Rates.

| Metric | Clean Scenario (30dB) | Noisy Scenario (5dB) | Overlap Scenario (15dB) |
| :--- | :--- | :--- | :--- |
| **SSL Error (RMSAE)** | 1.2° | 7.4° | 18.5° |
| **Beamforming (SDRi)** | +1.9 dB | +10.2 dB | +1.5 dB |
| **Denoising (PESQ)** | 4.30 | 3.25 | Bypassed (Preserved) |
| **Diarization (DER)** | 1.1% | 2.8% | 6.5% |
| **Transcription (WER)** | 7.8% | 13.5% | 19.8% |
| **Wall Clock Latency** | ~454ms | ~510ms | ~540ms |

### Key Technological Integrations:
- **INT8 Whisper Quantization:** The transformer encoder bounds dynamically down from FP32 parameters pushing model memory heavily toward $\sim$20 MB, enabling global real-time $<500$ms inference tracking gracefully.
- **Daubechies-4 Wavelet Packet Decomposition (WPD):** Replaces legacy asymmetrical standard DWT bins. WPD perfectly slices signal representation into 8 discrete 1kHz boundaries symmetrically. This accurately preserves $2-4$kHz textual formants exactly saving WER interpretation heavily vs destructive algorithms.
- **Wavelet Pre-Denoising (SSL):** Explicit Single-Level Soft Thresholding applied precisely prior to standard Cross-Correlation Arrays structurally scrubbing polynomial reverb equations natively returning Spatial Accuracy gracefully under tight multi-talk conditions.
- **Adaptive EMA VAD Tracking:** Moving Average updates seamlessly track changing room ambiance avoiding explicit continuous tuning completely correctly limiting baseline variables implicitly!

---

## Project Structure

```
Wavelets/
├── docs/                                  # Final Academic PDF & latex parameters
├── research_paper_images/                 # IEEE formatted final visualization maps
│
├── edge_audio_intelligence/               # Full modular Edge pipeline
│   ├── modules/                           # Final implemented stages
│   │   ├── ssl/                           #   Wavelet Pre-denoised GCC-PHAT
│   │   ├── beamforming/                   #   EMA-gated MVDR 
│   │   ├── enhancement/                   #   Db4 Wavelet packet decomposition (WPD)
│   │   ├── asr/                           #   INT8 Quantized Whisper API
│   │   ├── separation/                    #   Speaker separation bounds
│   │   └── diarization/                   #   Dynamic Pyannote processing natively
│   │
│   ├── backend/                           # FastAPI Service Infrastructure
│   ├── frontend/                          # Interactive React Application GUI
│   │
│   ├── testbench/                         # Acoustic simulation lab (Pyroomacoustics ISM)
│   ├── experiments/                       # Target bench sequences naturally testing constraints
│   └── pipeline/                          # System integration
```

---

## 🚀 Quick Start (Simulation & Dashboard)

### 1. Unified Environment Setup

```bash
# Clone / navigate to project
cd /path/to/Wavelets

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r edge_audio_intelligence/requirements.txt
```

### 2. Booting The Interactive Application Server

The project ships natively bridging its complex algorithms directly via an interconnected React + FastAPI Dashboard seamlessly mapping environments visually!

**Start the FastAPI Backend Logic:**
```bash
cd edge_audio_intelligence/backend
# (Ensure your venv is active)
uvicorn main:app --reload --port 8000
```

**Start the React Frontend GUI:**
*(In a separate terminal window)*
```bash
cd edge_audio_intelligence/frontend

# Install react constraints explicitly natively
npm install

# Start Vite GUI Server
npm run dev
```
Navigate perfectly to the `localhost` URL provided dynamically utilizing visually mapped Scene Layout arrays explicitly tracking audio bounds safely through unified stages linearly!

---

## 🧑‍💻 Executing The Offline Benchmarking Tests

If you wish to test exactly the rigorous benchmark evaluations driving the results specifically via console array evaluations natively:

```bash
# Run the Final V2 Optimized Sub-pipeline
PYTHONPATH=. python edge_audio_intelligence/experiments/05_end_to_end_pipeline.py

# Evaluate Diarization Metrics Exclusively
PYTHONPATH=. python edge_audio_intelligence/experiments/04_diarization_comparison.py
```

Results naturally parse and evaluate logically into the `/results/` path utilizing `matplotlib` generation natively verifying matrix tracking exactly.

---

## Evaluating Signal Traces & Visualizations

Our unified mathematical visual trackers cleanly generate visual parameters directly mapping performance inherently:

```python
from edge_audio_intelligence.testbench.visualizer import plot_scene_layout, plot_pipeline_signals

# Plot spatial tracking room arrays accurately mapping Array locations natively
plot_scene_layout(scene, save_path="results/figures/scene_layout.png")
```

All formal verified graphic traces explicitly charting Word Error drops implicitly vs Standard tracking exactly output towards `../research_paper_images/` natively supporting the LaTeX document completely natively gracefully.
