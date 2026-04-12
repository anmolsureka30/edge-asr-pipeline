"""
Profiling utilities: latency, FLOPs, MACs, memory, energy estimation.

Provides measurement and estimation tools for edge deployment analysis.

Key metrics for edge audio (in order of importance):
1. Latency (ms)          — measured via perf_counter
2. RTF (real-time factor) — latency / audio_duration
3. MACs                   — multiply-accumulate operations (1 MAC = 2 FLOPs)
4. Peak Memory (KB)       — via tracemalloc
5. Parameters             — model weight count
6. Energy (mJ)            — estimated as TDP_watts * latency_seconds * 1000
7. MACs/second            — throughput: MACs / latency
8. Memory Bandwidth (MB/s) — peak_memory / latency

Reference power consumption for edge devices (Thermal Design Power):
    Raspberry Pi 4B:    5.0 W (active), 1.0 W (idle)
    Jetson Nano:       10.0 W (active), 2.0 W (idle)
    Phone CPU:          2.0 W (active), 0.3 W (idle)
    Phone NPU:          0.5 W (active), 0.05 W (idle)
    DSP (always-on):    0.005 W (5 mW)

Source: Cross3D-Edge (Yin & Verhelst 2025), Ambiq Atomiq specs,
        AONDevices AON1120 specs.
"""

import time
import logging
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Reference TDP values for energy estimation (watts)
DEVICE_TDP = {
    "raspberry_pi_4b": 5.0,
    "jetson_nano": 10.0,
    "phone_cpu": 2.0,
    "phone_npu": 0.5,
    "dsp_always_on": 0.005,
    "generic_cpu": 15.0,  # Laptop/desktop CPU (single core estimate)
}


@dataclass
class ComputeProfile:
    """Complete compute profile for a pipeline module.

    All the metrics needed to evaluate edge deployment feasibility.
    """
    module_name: str = ""

    # Measured
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    peak_memory_kb: float = 0.0

    # Counted/estimated
    parameters: int = 0
    macs: int = 0           # Multiply-Accumulate Operations
    flops: int = 0          # Floating Point Operations (= 2 * MACs for most ops)

    # Derived
    rtf: float = 0.0                # Real-Time Factor
    energy_mj: float = 0.0          # Estimated energy in millijoules
    macs_per_second: float = 0.0    # Throughput
    memory_bandwidth_mbps: float = 0.0  # Memory bandwidth in MB/s
    device: str = "generic_cpu"     # Device used for energy estimation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "latency_ms": round(self.latency_ms, 2),
            "latency_std_ms": round(self.latency_std_ms, 2),
            "peak_memory_kb": round(self.peak_memory_kb, 1),
            "parameters": self.parameters,
            "macs": self.macs,
            "flops": self.flops,
            "rtf": round(self.rtf, 4),
            "energy_mj": round(self.energy_mj, 3),
            "macs_per_second": round(self.macs_per_second, 0),
            "memory_bandwidth_mbps": round(self.memory_bandwidth_mbps, 2),
            "device": self.device,
        }

    def summary(self) -> str:
        lines = [f"--- {self.module_name} ---"]
        lines.append(f"  Latency: {self.latency_ms:.1f} +/- {self.latency_std_ms:.1f} ms")
        lines.append(f"  RTF: {self.rtf:.4f}")
        lines.append(f"  Parameters: {self.parameters:,}")
        if self.macs > 0:
            lines.append(f"  MACs: {_format_ops(self.macs)}")
            lines.append(f"  FLOPs: {_format_ops(self.flops)}")
            lines.append(f"  Throughput: {_format_ops(self.macs_per_second)}/s")
        lines.append(f"  Peak Memory: {self.peak_memory_kb:.0f} KB")
        lines.append(f"  Energy ({self.device}): {self.energy_mj:.2f} mJ")
        if self.memory_bandwidth_mbps > 0:
            lines.append(f"  Memory BW: {self.memory_bandwidth_mbps:.1f} MB/s")
        return "\n".join(lines)


def _format_ops(ops: float) -> str:
    """Format operation count with SI prefix."""
    if ops >= 1e9:
        return f"{ops/1e9:.2f}G"
    elif ops >= 1e6:
        return f"{ops/1e6:.2f}M"
    elif ops >= 1e3:
        return f"{ops/1e3:.1f}K"
    return f"{ops:.0f}"


@contextmanager
def timer():
    """Context manager that yields a dict with 'elapsed_ms' after exit."""
    result = {}
    t0 = time.perf_counter()
    yield result
    t1 = time.perf_counter()
    result["elapsed_ms"] = (t1 - t0) * 1000.0


@contextmanager
def memory_tracker():
    """Context manager that yields a dict with 'peak_memory_kb' after exit."""
    result = {}
    tracemalloc.start()
    yield result
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_memory_kb"] = peak / 1024.0


def estimate_energy_mj(
    latency_ms: float,
    device: str = "generic_cpu",
) -> float:
    """Estimate energy consumption in millijoules.

    Energy = Power (W) * Time (s) * 1000 (to get mJ)

    This is an upper-bound estimate using device TDP.
    Actual energy is lower (not all circuits active).
    For precise measurement, use Joulescope or Monsoon power monitor.

    Args:
        latency_ms: Processing time in milliseconds.
        device: Device name from DEVICE_TDP.

    Returns:
        Estimated energy in millijoules.
    """
    tdp = DEVICE_TDP.get(device, DEVICE_TDP["generic_cpu"])
    return tdp * (latency_ms / 1000.0) * 1000.0  # W * s * 1000 = mJ


def estimate_macs_for_fft(n_fft: int, n_frames: int = 1) -> int:
    """Estimate MACs for FFT computation.

    FFT has O(N/2 * log2(N)) complex multiplications.
    Each complex multiply = 4 real MACs.

    Args:
        n_fft: FFT size.
        n_frames: Number of frames.

    Returns:
        Estimated MACs.
    """
    log2_n = int(np.ceil(np.log2(n_fft)))
    macs_per_frame = int((n_fft / 2) * log2_n * 4)
    return macs_per_frame * n_frames


def estimate_macs_for_convolution(signal_len: int, kernel_len: int) -> int:
    """Estimate MACs for 1D convolution (direct or overlap-add FFT).

    Direct: N * K MACs (N = signal length, K = kernel length)
    FFT-based: 2 * FFT(N+K) + N+K pointwise multiply

    We report the FFT-based estimate since scipy.fftconvolve is used.

    Args:
        signal_len: Input signal length.
        kernel_len: Convolution kernel length.

    Returns:
        Estimated MACs.
    """
    conv_len = signal_len + kernel_len - 1
    n_fft = int(2 ** np.ceil(np.log2(conv_len)))
    # 2 forward FFTs + 1 inverse FFT + pointwise multiply
    fft_macs = 3 * estimate_macs_for_fft(n_fft)
    pointwise_macs = n_fft * 4  # complex multiply
    return fft_macs + pointwise_macs


def estimate_macs_for_dwt(signal_len: int, levels: int, filter_len: int = 6) -> int:
    """Estimate MACs for multi-level DWT.

    Each DWT level: 2 convolutions (lowpass + highpass) of length N/2^(j-1)
    with filter of length filter_len.

    Args:
        signal_len: Input signal length.
        levels: Number of DWT decomposition levels.
        filter_len: Wavelet filter length.

    Returns:
        Estimated MACs.
    """
    total = 0
    n = signal_len
    for j in range(levels):
        # Two convolutions (lowpass, highpass) at this level
        total += 2 * n * filter_len
        n = n // 2  # Downsample by 2
    return total


def estimate_macs_for_whisper(model_size: str, audio_duration_s: float) -> int:
    """Estimate MACs for Whisper inference.

    Based on published analysis of Whisper architecture:
    - Encoder: ~80% of compute (self-attention + FFN)
    - Decoder: ~20% of compute (cross-attention + generation)

    Rough estimates per 30s audio chunk:
        tiny:  ~1.5 GMACs
        base:  ~3.0 GMACs
        small: ~10  GMACs
        medium: ~30 GMACs

    Source: "Quantization for OpenAI's Whisper Models" (2025)

    Args:
        model_size: 'tiny', 'base', 'small', 'medium'.
        audio_duration_s: Audio duration in seconds.

    Returns:
        Estimated MACs.
    """
    # MACs per 30-second chunk (approximate)
    macs_per_30s = {
        "tiny": 1_500_000_000,
        "base": 3_000_000_000,
        "small": 10_000_000_000,
        "medium": 30_000_000_000,
        "large": 100_000_000_000,
    }
    base_macs = macs_per_30s.get(model_size, macs_per_30s["base"])
    # Scale linearly with audio duration
    return int(base_macs * (audio_duration_s / 30.0))


def profile_module(
    module: Any,
    data: Dict[str, Any],
    n_runs: int = 50,
    device: str = "generic_cpu",
) -> ComputeProfile:
    """Full compute profile for a pipeline module.

    Measures latency, memory, and estimates MACs, energy, throughput.

    Args:
        module: A BaseModule instance.
        data: Input data dictionary.
        n_runs: Number of runs for latency averaging.
        device: Target device for energy estimation.

    Returns:
        ComputeProfile with all metrics.
    """
    # Latency measurement
    module.process(data.copy())  # Warm-up

    times = []
    for _ in range(n_runs):
        data_copy = data.copy()
        t0 = time.perf_counter()
        module.process(data_copy)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    latency_ms = float(np.mean(times))
    latency_std = float(np.std(times))

    # Memory measurement
    with memory_tracker() as m:
        module.process(data.copy())

    # FLOPs, MACs, parameters
    flops = module.estimate_flops(data)
    macs = flops // 2 if flops > 0 else 0  # 1 MAC = 2 FLOPs
    params = module.count_parameters()

    # Audio duration for RTF
    sr = data.get("sample_rate", 16000)
    audio_dur_s = 0.0
    if "multichannel_audio" in data:
        audio_dur_s = data["multichannel_audio"].shape[-1] / sr
    elif "enhanced_audio" in data:
        audio = data["enhanced_audio"]
        if isinstance(audio, np.ndarray):
            audio_dur_s = audio.shape[-1] / sr

    rtf = (latency_ms / 1000.0) / max(audio_dur_s, 1e-6)

    # Derived metrics
    energy_mj = estimate_energy_mj(latency_ms, device)
    macs_per_sec = macs / max(latency_ms / 1000.0, 1e-6) if macs > 0 else 0
    mem_bw = (m["peak_memory_kb"] / 1024.0) / max(latency_ms / 1000.0, 1e-6)  # MB/s

    profile = ComputeProfile(
        module_name=module.name,
        latency_ms=latency_ms,
        latency_std_ms=latency_std,
        peak_memory_kb=m["peak_memory_kb"],
        parameters=params,
        macs=macs,
        flops=flops,
        rtf=rtf,
        energy_mj=energy_mj,
        macs_per_second=macs_per_sec,
        memory_bandwidth_mbps=mem_bw,
        device=device,
    )

    logger.info(f"\n{profile.summary()}")
    return profile


def profile_pipeline(
    pipeline: Any,
    data: Dict[str, Any],
    device: str = "generic_cpu",
) -> Dict[str, Any]:
    """Profile an entire pipeline, returning per-module and total metrics.

    Args:
        pipeline: A CascadePipeline instance.
        data: Input data dictionary.
        device: Target device for energy estimation.

    Returns:
        Dict with 'modules' (list of ComputeProfile dicts) and 'total' summary.
    """
    profiles = []
    total_macs = 0
    total_energy = 0.0
    total_latency = 0.0
    total_params = 0
    total_memory = 0.0

    for module in pipeline.modules:
        profile = profile_module(module, data, n_runs=10, device=device)
        profiles.append(profile.to_dict())

        total_macs += profile.macs
        total_energy += profile.energy_mj
        total_latency += profile.latency_ms
        total_params += profile.parameters
        total_memory = max(total_memory, profile.peak_memory_kb)

        # Update data for next module
        data = module.process(data.copy())

    sr = data.get("sample_rate", 16000)
    audio_dur = 0.0
    if "multichannel_audio" in data:
        audio_dur = data["multichannel_audio"].shape[-1] / sr

    total = {
        "total_latency_ms": round(total_latency, 1),
        "total_macs": total_macs,
        "total_macs_formatted": _format_ops(total_macs),
        "total_flops": total_macs * 2,
        "total_flops_formatted": _format_ops(total_macs * 2),
        "total_energy_mj": round(total_energy, 2),
        "total_parameters": total_params,
        "total_rtf": round(total_latency / 1000.0 / max(audio_dur, 1e-6), 4),
        "peak_memory_kb": round(total_memory, 0),
        "device": device,
    }

    return {"modules": profiles, "total": total}
