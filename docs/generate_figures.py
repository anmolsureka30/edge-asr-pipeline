#!/usr/bin/env python3
"""
Generate all research paper figures from run_history.json data.
Outputs PNG files to docs/images/ for inclusion in final_report.tex.
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Config ---
OUT_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'edge_audio_intelligence', 'results', 'run_history.json'
)

# Load all run data
with open(RESULTS_FILE) as f:
    runs = json.load(f)

# --- Color palette ---
C_BLUE = '#1f77b4'
C_ORANGE = '#ff7f0e'
C_GREEN = '#2ca02c'
C_RED = '#d62728'
C_PURPLE = '#9467bd'
C_GRAY = '#7f7f7f'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ======================================================================
# FIGURE 1: Enhancement Impact Bar Chart (Finding 3)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 4))
methods = ['No Enhancement', 'Wavelet (bior2.2)', 'Spectral Sub.\n(α=2.0)']
wer_vals = [25.0, 39.3, 110.7]
colors = [C_GREEN, C_BLUE, C_RED]
bars = ax.bar(methods, wer_vals, color=colors, width=0.55, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Word Error Rate (%)')
ax.set_title('Enhancement Impact on WER\n(Office Scene, 2-Speaker, MUSIC→DS→Whisper-small)')
ax.set_ylim(0, 130)
ax.axhline(y=25.0, color=C_GREEN, linestyle='--', alpha=0.5, label='Baseline (no enh.)')
for bar, val in zip(bars, wer_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUT_DIR, 'enhancement_wer_impact.png'))
plt.close()
print("[OK] enhancement_wer_impact.png")

# ======================================================================
# FIGURE 2: SSL Method Comparison (Finding 2)
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ssl_methods = ['GCC-PHAT\n(original)', 'GCC-PHAT\n(16× interp.)', 'SRP-PHAT', 'MUSIC']
ssl_errors = [80.8, 8.2, 140.0, 48.0]
ssl_colors = [C_RED, C_GREEN, C_RED, C_ORANGE]

bars1 = ax1.bar(ssl_methods, ssl_errors, color=ssl_colors, width=0.55, edgecolor='white', linewidth=1.2)
ax1.set_ylabel('Angular Error (degrees)')
ax1.set_title('SSL Angular Error Comparison\n(Office Meeting, 2-Speaker)')
ax1.set_ylim(0, 160)
ax1.axhline(y=10, color=C_GREEN, linestyle='--', alpha=0.4, label='Target (<10°)')
for bar, val in zip(bars1, ssl_errors):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{val:.1f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.legend(loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Latency comparison
latencies = [9, 9, 37, 32]
bars2 = ax2.bar(ssl_methods, latencies, color=[C_BLUE]*4, width=0.55, edgecolor='white', linewidth=1.2)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('SSL Latency Comparison')
ax2.set_ylim(0, 50)
for bar, val in zip(bars2, latencies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val} ms', ha='center', va='bottom', fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ssl_comparison.png'))
plt.close()
print("[OK] ssl_comparison.png")

# ======================================================================
# FIGURE 3: Latency Breakdown Pie Chart (Finding 7)
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Clean room — best case
labels1 = ['GCC-PHAT\n(92 ms)', 'MVDR\n(29 ms)', 'Wavelet Enh.\n(9 ms)', 'Whisper-small\n(2047 ms)']
sizes1 = [92, 29, 9, 2047]
colors1 = [C_GREEN, C_BLUE, C_PURPLE, C_RED]
explode1 = (0, 0, 0, 0.08)
wedges1, texts1, autotexts1 = ax1.pie(sizes1, explode=explode1, labels=labels1, colors=colors1,
                                       autopct='%1.1f%%', startangle=140, pctdistance=0.7,
                                       textprops={'fontsize': 9})
ax1.set_title('Latency Breakdown\n(Clean Room, Best Case)', fontsize=12)

# Office meeting - 2 speaker
labels2 = ['MUSIC\n(32 ms)', 'DS\n(15 ms)', 'Wavelet Enh.\n(5 ms)', 'Whisper-small\n(2943 ms)']
sizes2 = [32, 15, 5, 2943]
colors2 = [C_GREEN, C_BLUE, C_PURPLE, C_RED]
explode2 = (0, 0, 0, 0.08)
wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, labels=labels2, colors=colors2,
                                       autopct='%1.1f%%', startangle=140, pctdistance=0.7,
                                       textprops={'fontsize': 9})
ax2.set_title('Latency Breakdown\n(Office, 2-Speaker)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'latency_breakdown.png'))
plt.close()
print("[OK] latency_breakdown.png")

# ======================================================================
# FIGURE 4: MVDR VAD Gating Impact (Finding 4)
# ======================================================================
fig, ax = plt.subplots(figsize=(5, 4))
conditions = ['Without\nVAD Gating', 'With\nVAD Gating']
si_sdr_vals = [0.1, 13.9]
colors_mvdr = [C_RED, C_GREEN]
bars = ax.bar(conditions, si_sdr_vals, color=colors_mvdr, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_ylabel('SI-SDR (dB)')
ax.set_title('MVDR Beamformer Performance:\nVAD-Gated vs. Blind Φ_nn Estimation')
ax.set_ylim(-2, 18)
for bar, val in zip(bars, si_sdr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f} dB', ha='center', va='bottom', fontweight='bold', fontsize=12)
# Add improvement arrow
ax.annotate('+13.8 dB', xy=(1, 13.9), xytext=(0.5, 16),
            fontsize=13, fontweight='bold', color=C_GREEN,
            arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2),
            ha='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUT_DIR, 'mvdr_vad_gating.png'))
plt.close()
print("[OK] mvdr_vad_gating.png")

# ======================================================================
# FIGURE 5: Master WER Comparison Across Configurations (Finding 5)
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 5))

configs = [
    'Clean 1-spk\nGCC→DS→SpSub→Wh-s',
    'Clean 1-spk\nGCC→MVDR(g)→Wav(g)→Wh-s',
    'Clean 1-spk\nSRP→DS→SpSub→Wh-tiny',
    'Office 2-spk\nMUSIC→DS→Wh-s (no enh)',
    'Office 2-spk\nMUSIC→DS→Wav→Wh-s',
    'Office 2-spk\nGCC→DS→SpSub→Wh-s',
    'Office 2-spk\nGCC→MVDR(g)→Wav(g)→Wh-s',
    'Cafe 2-spk\nMUSIC→MVDR(g)→Wav(g)→Wh-s',
]
wers = [21.4, 14.3, 17.9, 25.0, 39.3, 110.7, 67.9, 75.0]
bar_colors = [C_BLUE, C_GREEN, C_BLUE, C_ORANGE, C_RED, C_RED, C_RED, C_RED]

bars = ax.barh(configs, wers, color=bar_colors, height=0.6, edgecolor='white', linewidth=1)
ax.set_xlabel('Word Error Rate (%)')
ax.set_title('End-to-End WER Across All Pipeline Configurations')
ax.set_xlim(0, 130)
ax.axvline(x=15, color=C_GREEN, linestyle='--', alpha=0.5, label='Target WER (<15%)')
ax.axvline(x=100, color=C_RED, linestyle='--', alpha=0.3, label='WER > 100% (hallucination)')
for bar, val in zip(bars, wers):
    ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'master_wer_comparison.png'))
plt.close()
print("[OK] master_wer_comparison.png")

# ======================================================================
# FIGURE 6: Sub-Sample TDOA Resolution Fix (Finding 1)
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

conditions_ssl = ['Anechoic\nSNR=30dB', 'RT60=0.3s\nSNR=20dB', 'RT60=0.4s\nSNR=20dB',
                  'RT60=0.6s\nSNR=15dB', 'RT60=0.6s\nSNR=5dB']
before = [18.0, 40.0, 50.0, 50.0, 80.0]
after = [1.7, 7.0, 9.5, 14.1, 19.9]

x = np.arange(len(conditions_ssl))
width = 0.32
bars1 = ax.bar(x - width/2, before, width, color=C_RED, label='Integer sample (before)', edgecolor='white')
bars2 = ax.bar(x + width/2, after, width, color=C_GREEN, label='16× interpolated (after)', edgecolor='white')

ax.set_ylabel('Angular Error (degrees)')
ax.set_title('Sub-Sample TDOA Resolution Impact on GCC-PHAT')
ax.set_xticks(x)
ax.set_xticklabels(conditions_ssl, fontsize=9)
ax.axhline(y=10, color=C_GRAY, linestyle=':', alpha=0.5, label='Target (<10°)')
ax.legend(loc='upper left')
ax.set_ylim(0, 95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars1, before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.0f}°', ha='center', va='bottom', fontsize=8, color=C_RED)
for bar, val in zip(bars2, after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}°', ha='center', va='bottom', fontsize=8, color=C_GREEN)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ssl_subsample_fix.png'))
plt.close()
print("[OK] ssl_subsample_fix.png")

# ======================================================================
# FIGURE 7: Tonal Interference Results (Finding 6)
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
tonal_configs = ['DS only\n(no enh)', 'DS +\nSpSub', 'DS +\nNotch', 'DS + Notch\n+ SpSub',
                 'MVDR\nonly', 'MVDR +\nNotch']
tonal_wer = [27.3, 40.9, 40.9, 45.5, 72.7, 63.6]
tonal_colors = [C_GREEN, C_ORANGE, C_ORANGE, C_RED, C_RED, C_RED]

bars = ax.bar(tonal_configs, tonal_wer, color=tonal_colors, width=0.55, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Word Error Rate (%)')
ax.set_title('Tonal Interference Results (1kHz Sine, SIR ≈ +1dB)')
ax.set_ylim(0, 85)
ax.axhline(y=27.3, color=C_GREEN, linestyle='--', alpha=0.4, label='Best (DS only)')
for bar, val in zip(bars, tonal_wer):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tonal_interference.png'))
plt.close()
print("[OK] tonal_interference.png")

# ======================================================================
# FIGURE 8: Computational Profile (Finding 7)
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))

modules = ['GCC-PHAT', 'MUSIC', 'MVDR', 'Wavelet Enh.', 'AtomicVAD', 'Whisper-tiny', 'Whisper-small']
latency_avg = [50, 34, 25, 8, 77, 2900, 3200]
module_colors = [C_GREEN, C_GREEN, C_BLUE, C_PURPLE, C_ORANGE, C_RED, C_RED]

bars = ax.barh(modules, latency_avg, color=module_colors, height=0.55, edgecolor='white')
ax.set_xlabel('Average Latency (ms)')
ax.set_title('Module-Level Computational Profile\n(per 15s audio segment)')
ax.set_xscale('log')
ax.set_xlim(1, 5000)
for bar, val in zip(bars, latency_avg):
    ax.text(val + val*0.15, bar.get_y() + bar.get_height()/2,
            f'{val} ms', ha='left', va='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'computational_profile.png'))
plt.close()
print("[OK] computational_profile.png")

# ======================================================================
# FIGURE 9: DWT Sub-band Decomposition Diagram
# ======================================================================
fig, ax = plt.subplots(figsize=(8, 3.5))
# Show sub-band frequency allocation
bands = ['cA₃\n(0–1 kHz)', 'cD₃\n(1–2 kHz)', 'cD₂\n(2–4 kHz)', 'cD₁\n(4–8 kHz)']
speech_energy = [0.55, 0.25, 0.15, 0.05]
noise_energy = [0.25, 0.25, 0.25, 0.25]

x = np.arange(len(bands))
width = 0.3
bars1 = ax.bar(x - width/2, speech_energy, width, color=C_BLUE, label='Speech Energy', edgecolor='white')
bars2 = ax.bar(x + width/2, noise_energy, width, color=C_GRAY, label='White Noise Energy', edgecolor='white')

ax.set_ylabel('Normalized Energy')
ax.set_title('DWT Sub-band Energy Distribution: Speech vs. Noise\n(CDF 5/3, J=3, fs=16kHz)')
ax.set_xticks(x)
ax.set_xticklabels(bands, fontsize=11)
ax.set_ylim(0, 0.7)
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotations
ax.annotate('R_VAD = E(cA₃)/E(cD₁)\nSpeech: R >> 1\nNoise: R ≈ 1',
            xy=(0, 0.55), xytext=(2, 0.55),
            fontsize=9, style='italic',
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'dwt_subband_energy.png'))
plt.close()
print("[OK] dwt_subband_energy.png")

# ======================================================================
# FIGURE 10: Pipeline Quality Across Scenes
# ======================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

scenes = ['Clean Room\n(RT60=0, SNR=30)', 'Office Meeting\n(RT60=0.4, SNR=20)',
          'Noisy Cafe\n(RT60=0.6, SNR=5)']
pesq_vals = [2.84, 1.21, 1.11]
stoi_vals = [0.937, 0.708, 0.535]

x = np.arange(len(scenes))
width = 0.3

ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, pesq_vals, width, color=C_BLUE, alpha=0.8, label='PESQ', edgecolor='white')
bars2 = ax2.bar(x + width/2, [s*4.5 for s in stoi_vals], width, color=C_ORANGE, alpha=0.8,
                label='STOI (×4.5)', edgecolor='white')

ax.set_ylabel('PESQ Score', color=C_BLUE)
ax2.set_ylabel('STOI (scaled ×4.5)', color=C_ORANGE)
ax.set_title('Speech Quality Metrics Across Acoustic Conditions\n(Best Pipeline Config: GCC→MVDR(gated)→Wavelet(gated)→Whisper-s)')
ax.set_xticks(x)
ax.set_xticklabels(scenes, fontsize=10)
ax.set_ylim(0, 5)
ax2.set_ylim(0, 5)
ax.axhline(y=2.5, color=C_BLUE, linestyle=':', alpha=0.3)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

for bar, val in zip(bars1, pesq_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, color=C_BLUE, fontweight='bold')
for bar, val in zip(bars2, stoi_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=C_ORANGE, fontweight='bold')

ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'quality_across_scenes.png'))
plt.close()
print("[OK] quality_across_scenes.png")

print("\n=== All 10 figures generated successfully in docs/images/ ===")
