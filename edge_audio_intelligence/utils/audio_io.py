"""
Audio I/O utilities: load, save, resample audio files.

Supports WAV, FLAC, and other formats via soundfile/librosa.
All internal processing uses 16 kHz mono float32.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_SR = 16000


def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = DEFAULT_SR,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and optionally resample.

    Args:
        path: Path to audio file (WAV, FLAC, etc.).
        sr: Target sample rate. None keeps original rate.
        mono: If True, mix to mono.

    Returns:
        Tuple of (audio array [n_samples] or [n_channels, n_samples], sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # audio shape: [n_samples, n_channels]

    if mono and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    # Transpose to [n_channels, n_samples]
    audio = audio.T

    if mono:
        audio = audio[0]  # [n_samples]

    if sr is not None and file_sr != sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr

    logger.debug(f"Loaded {path.name}: shape={audio.shape}, sr={file_sr}")
    return audio, file_sr


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
) -> None:
    """Save audio array to file.

    Args:
        path: Output file path.
        audio: Audio array [n_samples] or [n_channels, n_samples].
        sr: Sample rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim == 2:
        # [n_channels, n_samples] -> [n_samples, n_channels]
        audio = audio.T

    sf.write(str(path), audio, sr)
    logger.debug(f"Saved audio to {path}: sr={sr}")


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio using scipy.

    Args:
        audio: Input audio [n_samples] or [n_channels, n_samples].
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio

    from scipy.signal import resample_poly
    import math

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float32)
    else:
        return np.array(
            [resample_poly(ch, up, down) for ch in audio], dtype=np.float32
        )
