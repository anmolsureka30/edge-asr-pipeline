"""
Room Impulse Response (RIR) Generation.

Mathematical background:
    A Room Impulse Response h(t) encodes the complete acoustic path
    from a sound source to a microphone, including:
    - Direct path (line-of-sight)
    - Wall reflections (first-order, second-order, ...)
    - Ceiling/floor reflections
    - Attenuation with distance (1/d falloff)
    - Absorption at each surface (reflection coefficient beta)

    The RIR is a sum of delayed, attenuated impulses:

        h(t) = sum_k  a_k * delta(t - tau_k)

    where for each path k:
        tau_k = d_k / c          (propagation delay)
        a_k   = (1/d_k) * prod(beta_i)  (attenuation)

        d_k   = total path length of path k (metres)
        c     = speed of sound (343 m/s)
        beta_i = reflection coefficient of i-th wall hit (0 < beta < 1)

    The observed signal at mic m from source s is then:

        x_m(t) = s(t) * h_ms(t)
               = sum_k  a_k * s(t - tau_k)

    This is the convolution of the clean source with the RIR — the room
    acts as a delay-and-scale machine.

    For multiple sources with additive noise:
        x_m(t) = sum_s [ h_ms(t) * s_s(t) ] + n_m(t)

    Ref: PIPELINE_ALGORITHM.md Section 0 (Signal Model), Eq. 0.1-0.3

Implementation uses pyroomacoustics Image Source Method (ISM) which
computes image sources (virtual mirrors of the real source across each
wall) up to max_order reflections. Each image source contributes one
impulse to the RIR.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def sabine_rt60(
    volume: float,
    surface_area: float,
    absorption_coeff: float,
) -> float:
    """Compute RT60 using Sabine's equation.

    RT60 = 0.161 * V / (S * alpha)

    where:
        V = room volume (m^3)
        S = total surface area (m^2)
        alpha = average absorption coefficient

    Args:
        volume: Room volume in cubic metres.
        surface_area: Total wall/ceiling/floor surface area in m^2.
        absorption_coeff: Average absorption coefficient (0 to 1).

    Returns:
        RT60 in seconds.
    """
    if absorption_coeff <= 0:
        return float("inf")
    return 0.161 * volume / (surface_area * absorption_coeff)


def absorption_from_rt60(
    rt60: float,
    room_dim: List[float],
) -> Tuple[float, int]:
    """Compute wall absorption coefficient and max reflection order from RT60.

    Uses pyroomacoustics inverse_sabine for accurate ISM parameters.

    Args:
        rt60: Target reverberation time in seconds.
        room_dim: [length, width, height] in metres.

    Returns:
        Tuple of (energy_absorption_coeff, max_order).

    Raises:
        ValueError: If RT60 is too small for the room size.
    """
    import pyroomacoustics as pra
    return pra.inverse_sabine(rt60, room_dim)


class RIRGenerator:
    """Generates Room Impulse Responses using the Image Source Method.

    This class wraps pyroomacoustics to generate RIRs for arbitrary
    room geometries, source positions, and microphone positions.

    The generated RIR h_ms[n] can then be convolved with any source
    signal to produce realistic reverberant audio.

    Usage:
        gen = RIRGenerator()
        rirs, metadata = gen.generate(
            room_dim=[6, 5, 3],
            source_positions=[[2, 3.5, 1.5]],
            mic_positions=[[3, 2.5, 1.2], [3.015, 2.5, 1.2]],
            rt60=0.4,
            fs=16000,
        )
        # rirs[mic_idx][source_idx] = h_ms as numpy array
        # Use convolve_rir(signal, rirs[m][s]) to apply
    """

    def __init__(self, speed_of_sound: float = 343.0):
        self.speed_of_sound = speed_of_sound

    def generate(
        self,
        room_dim: List[float],
        source_positions: List[List[float]],
        mic_positions: List[List[float]],
        rt60: float = 0.3,
        fs: int = 16000,
        air_absorption: bool = True,
    ) -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
        """Generate RIRs for all source-mic pairs.

        Args:
            room_dim: [length, width, height] in metres.
            source_positions: List of [x, y, z] source positions.
            mic_positions: List of [x, y, z] mic positions.
            rt60: Reverberation time in seconds. 0 = anechoic.
            fs: Sample rate in Hz.
            air_absorption: Whether to model air absorption.

        Returns:
            Tuple of (rirs, metadata):
                rirs[m][s] = np.ndarray RIR from source s to mic m
                metadata = dict with room info, max_order, absorption, etc.
        """
        import pyroomacoustics as pra

        # Compute room parameters
        if rt60 >= 0.2:
            try:
                e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
                room = pra.ShoeBox(
                    room_dim,
                    fs=fs,
                    materials=pra.Material(e_absorption),
                    max_order=max_order,
                    air_absorption=air_absorption,
                )
            except ValueError:
                logger.warning(
                    f"inverse_sabine failed for RT60={rt60}s, "
                    f"room={room_dim}. Using anechoic."
                )
                e_absorption, max_order = 1.0, 0
                room = pra.ShoeBox(room_dim, fs=fs, max_order=0)
        else:
            e_absorption, max_order = 1.0, 0
            room = pra.ShoeBox(room_dim, fs=fs, max_order=0)

        # Add sources (no signal needed — just computing RIRs)
        for pos in source_positions:
            room.add_source(pos)

        # Add microphones [3, n_mics]
        mic_array = np.array(mic_positions).T
        room.add_microphone_array(mic_array)

        # Compute RIRs
        room.compute_rir()

        # Extract: room.rir[mic_idx][source_idx] = np.ndarray
        rirs = room.rir  # List[List[np.ndarray]]

        metadata = {
            "room_dim": room_dim,
            "rt60": rt60,
            "fs": fs,
            "e_absorption": float(e_absorption),
            "max_order": int(max_order),
            "n_sources": len(source_positions),
            "n_mics": len(mic_positions),
            "speed_of_sound": self.speed_of_sound,
        }

        # Log RIR lengths
        for m in range(len(mic_positions)):
            for s in range(len(source_positions)):
                rir_len = len(rirs[m][s])
                rir_dur_ms = rir_len / fs * 1000
                logger.debug(
                    f"RIR[mic={m}][src={s}]: {rir_len} samples ({rir_dur_ms:.0f}ms)"
                )

        return rirs, metadata

    def generate_with_simulation(
        self,
        room_dim: List[float],
        source_signals: List[np.ndarray],
        source_positions: List[List[float]],
        mic_positions: List[List[float]],
        rt60: float = 0.3,
        fs: int = 16000,
    ) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray]:
        """Generate RIRs AND simulate the full multichannel output.

        Convenience method that does RIR generation + convolution in one step.

        Args:
            room_dim: [length, width, height].
            source_signals: List of clean source signals [n_samples].
            source_positions: List of [x, y, z] source positions.
            mic_positions: List of [x, y, z] mic positions.
            rt60: Reverberation time.
            fs: Sample rate.

        Returns:
            Tuple of (multichannel_audio, rirs, reverberant_per_source):
                multichannel_audio: [n_mics, n_samples] mixed output
                rirs: [mic][source] RIR arrays
                reverberant_per_source: [n_sources, n_mics, n_samples]
        """
        import pyroomacoustics as pra

        n_samples = max(len(s) for s in source_signals)

        # Build room
        if rt60 >= 0.2:
            try:
                e_abs, max_ord = pra.inverse_sabine(rt60, room_dim)
                room = pra.ShoeBox(
                    room_dim, fs=fs,
                    materials=pra.Material(e_abs),
                    max_order=max_ord,
                    air_absorption=True,
                )
            except ValueError:
                room = pra.ShoeBox(room_dim, fs=fs, max_order=0)
        else:
            room = pra.ShoeBox(room_dim, fs=fs, max_order=0)

        for pos, sig in zip(source_positions, source_signals):
            room.add_source(pos, signal=sig)

        room.add_microphone_array(np.array(mic_positions).T)
        room.simulate()

        multichannel = room.mic_array.signals
        if multichannel.shape[1] > n_samples:
            multichannel = multichannel[:, :n_samples]
        elif multichannel.shape[1] < n_samples:
            multichannel = np.pad(multichannel, ((0, 0), (0, n_samples - multichannel.shape[1])))

        # Extract per-source reverberant signals
        rirs = room.rir
        n_mics = len(mic_positions)
        n_src = len(source_signals)
        reverberant = np.zeros((n_src, n_mics, n_samples), dtype=np.float32)

        for s in range(n_src):
            for m in range(n_mics):
                h = rirs[m][s]
                conv = fftconvolve(source_signals[s], h, mode="full")
                length = min(len(conv), n_samples)
                reverberant[s, m, :length] = conv[:length]

        return multichannel.astype(np.float32), rirs, reverberant


def convolve_rir(
    signal: np.ndarray,
    rir: np.ndarray,
    trim_to: Optional[int] = None,
) -> np.ndarray:
    """Convolve a signal with a Room Impulse Response.

    x(t) = s(t) * h(t) = sum_k a_k * s(t - tau_k)

    Args:
        signal: Input signal [n_samples].
        rir: Room impulse response [rir_length].
        trim_to: If set, trim output to this length.

    Returns:
        Convolved signal.
    """
    result = fftconvolve(signal, rir, mode="full").astype(np.float32)
    if trim_to is not None and len(result) > trim_to:
        result = result[:trim_to]
    return result


def image_source_rir(
    room_dim: List[float],
    source_pos: List[float],
    mic_pos: List[float],
    fs: int = 16000,
    max_order: int = 10,
    absorption: float = 0.3,
) -> np.ndarray:
    """Compute RIR using Image Source Method directly (no pyroomacoustics).

    A pure-Python implementation for understanding. For production,
    use RIRGenerator which wraps pyroomacoustics (much faster).

    The Image Source Method works by:
    1. For each wall, create a mirror image of the source
    2. For each image source, the path length gives the delay,
       and the number of bounces gives the attenuation
    3. Sum all delayed impulses to get h(t)

    h(t) = sum_k (1/d_k) * beta^n_k * delta(t - d_k/c)

    Args:
        room_dim: [Lx, Ly, Lz] room dimensions.
        source_pos: [x, y, z] source position.
        mic_pos: [x, y, z] microphone position.
        fs: Sample rate.
        max_order: Maximum number of wall reflections.
        absorption: Wall absorption coefficient (0=perfect reflection, 1=full absorption).

    Returns:
        RIR as numpy array.
    """
    c = 343.0  # speed of sound
    beta = np.sqrt(1 - absorption)  # reflection coefficient

    Lx, Ly, Lz = room_dim
    sx, sy, sz = source_pos
    mx, my, mz = mic_pos

    # Collect all image source contributions
    delays = []
    amplitudes = []

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                # Image source position for reflection order (nx, ny, nz)
                # Even reflections: same position, odd: mirrored
                ix = (-1) ** nx * sx + nx * Lx if nx % 2 == 0 else (nx + 1) * Lx - sx + (nx - 1) * Lx
                # Simplified: use the standard ISM formula
                if nx % 2 == 0:
                    ix = sx + nx * Lx
                else:
                    ix = -sx + (nx + 1) * Lx if nx > 0 else -sx + nx * Lx

                if ny % 2 == 0:
                    iy = sy + ny * Ly
                else:
                    iy = -sy + (ny + 1) * Ly if ny > 0 else -sy + ny * Ly

                if nz % 2 == 0:
                    iz = sz + nz * Lz
                else:
                    iz = -sz + (nz + 1) * Lz if nz > 0 else -sz + nz * Lz

                # Distance from image source to microphone
                d = np.sqrt((ix - mx)**2 + (iy - my)**2 + (iz - mz)**2)

                if d < 1e-6:
                    continue

                # Number of reflections = |nx| + |ny| + |nz|
                n_bounces = abs(nx) + abs(ny) + abs(nz)

                # Attenuation: 1/d * beta^n_bounces
                a = (beta ** n_bounces) / d

                # Delay: d / c
                tau = d / c

                delays.append(tau)
                amplitudes.append(a)

    if not delays:
        return np.zeros(1, dtype=np.float32)

    # Convert to discrete RIR
    max_delay_samples = int(max(delays) * fs) + 1
    rir = np.zeros(max_delay_samples, dtype=np.float64)

    for tau, a in zip(delays, amplitudes):
        sample_idx = int(tau * fs)
        if 0 <= sample_idx < max_delay_samples:
            rir[sample_idx] += a

    return rir.astype(np.float32)
