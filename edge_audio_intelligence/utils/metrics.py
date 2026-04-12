"""
Metric implementations for all pipeline stages.

References: DATASETS_AND_METRICS.md, PIPELINE_ALGORITHM.md
Each metric includes the formula reference from the documentation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Localization metrics (DATASETS_AND_METRICS.md Section: Localization Metrics)
# ---------------------------------------------------------------------------

def angular_error(
    estimated_doa: np.ndarray,
    true_doa: np.ndarray,
) -> float:
    """Compute absolute angular error in degrees.

    For azimuth-only (scalar): simple circular distance min(|Δ|, 360-|Δ|).
    For azimuth+elevation (2D): spherical great-circle distance:
        arccos(sin(φ_t)sin(φ_e) + cos(φ_t)cos(φ_e)cos(θ_t - θ_e))
    Ref: ACOUSTIC_LAB.md Section 4.2

    Args:
        estimated_doa: Estimated direction [azimuth] or [azimuth, elevation] in degrees.
        true_doa: Ground truth direction in degrees.

    Returns:
        Angular error in degrees, in [0, 180].
    """
    est = np.atleast_1d(np.asarray(estimated_doa, dtype=np.float64))
    true = np.atleast_1d(np.asarray(true_doa, dtype=np.float64))

    # Azimuth-only case (1D or single value)
    if est.size == 1:
        diff = abs(float(est.flat[0]) - float(true.flat[0]))
        return min(diff, 360.0 - diff)

    # 2D case: [azimuth, elevation] — use great-circle distance
    if est.shape[-1] >= 2:
        az_est = np.radians(est[..., 0])
        el_est = np.radians(est[..., 1])
        az_true = np.radians(true[..., 0])
        el_true = np.radians(true[..., 1])

        cos_dist = (
            np.sin(el_true) * np.sin(el_est)
            + np.cos(el_true) * np.cos(el_est) * np.cos(az_true - az_est)
        )
        cos_dist = np.clip(cos_dist, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_dist)))

    # 1D array (just azimuth)
    diff = abs(float(est[0]) - float(true[0]))
    return min(diff, 360.0 - diff)


def rmsae(
    estimated_doas: np.ndarray,
    true_doas: np.ndarray,
) -> float:
    """Root Mean Square Angular Error over multiple frames.

    RMSAE = sqrt( (1/N) * sum(angular_error_i^2) )

    Args:
        estimated_doas: [n_frames, n_dims] array of estimated DOAs in degrees.
        true_doas: [n_frames, n_dims] array of true DOAs in degrees.

    Returns:
        RMSAE in degrees.
    """
    errors = []
    for est, true in zip(estimated_doas, true_doas):
        errors.append(angular_error(est, true))
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def position_error(
    estimated_pos: np.ndarray,
    true_pos: np.ndarray,
) -> float:
    """Euclidean distance between estimated and true source positions.

    Args:
        estimated_pos: [x, y] or [x, y, z] estimated position in meters.
        true_pos: [x, y] or [x, y, z] true position in meters.

    Returns:
        Distance in meters.
    """
    return float(np.linalg.norm(np.array(estimated_pos) - np.array(true_pos)))


# ---------------------------------------------------------------------------
# Enhancement metrics (DATASETS_AND_METRICS.md Section: Enhancement Metrics)
# ---------------------------------------------------------------------------

def pesq_score(
    ref: np.ndarray,
    deg: np.ndarray,
    sr: int = 16000,
    mode: str = "wb",
) -> float:
    """Perceptual Evaluation of Speech Quality (ITU-T P.862).

    Range: -0.5 to 4.5 (higher is better).

    Args:
        ref: Reference (clean) signal.
        deg: Degraded (enhanced) signal.
        sr: Sample rate (8000 or 16000).
        mode: 'nb' (narrowband) or 'wb' (wideband).

    Returns:
        PESQ score.
    """
    try:
        from pesq import pesq as compute_pesq
        # Ensure same length
        min_len = min(len(ref), len(deg))
        return float(compute_pesq(sr, ref[:min_len], deg[:min_len], mode))
    except ImportError:
        logger.warning("pesq package not installed. Run: pip install pesq")
        return float("nan")


def stoi_score(
    ref: np.ndarray,
    deg: np.ndarray,
    sr: int = 16000,
    extended: bool = False,
) -> float:
    """Short-Time Objective Intelligibility.

    Range: 0 to 1 (higher is better).

    Args:
        ref: Reference (clean) signal.
        deg: Degraded (enhanced) signal.
        sr: Sample rate.
        extended: If True, use extended STOI.

    Returns:
        STOI score.
    """
    try:
        from pystoi import stoi as compute_stoi
        min_len = min(len(ref), len(deg))
        return float(compute_stoi(ref[:min_len], deg[:min_len], sr, extended=extended))
    except ImportError:
        logger.warning("pystoi package not installed. Run: pip install pystoi")
        return float("nan")


def si_sdr(
    ref: np.ndarray,
    est: np.ndarray,
) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB).

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<est, ref> / ||ref||^2) * ref
    and e_noise = est - s_target

    Ref: PIPELINE_ALGORITHM.md Eq. 4.5

    Args:
        ref: Reference signal [n_samples].
        est: Estimated signal [n_samples].

    Returns:
        SI-SDR in dB.
    """
    min_len = min(len(ref), len(est))
    ref = ref[:min_len].astype(np.float64)
    est = est[:min_len].astype(np.float64)

    # Remove mean
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    # s_target = (<est, ref> / ||ref||^2) * ref
    dot = np.dot(est, ref)
    s_target = (dot / (np.dot(ref, ref) + 1e-10)) * ref

    # e_noise = est - s_target
    e_noise = est - s_target

    si_sdr_val = 10.0 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-10)
    )
    return float(si_sdr_val)


# ---------------------------------------------------------------------------
# ASR metrics (DATASETS_AND_METRICS.md Section: ASR Metrics)
# ---------------------------------------------------------------------------

def word_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    """Word Error Rate using edit distance.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=words in reference.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR output transcription.

    Returns:
        WER as a float (0.0 = perfect, >1.0 possible with many insertions).
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else float(m)

    # Dynamic programming for edit distance
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return float(dp[n][m]) / float(n)


def wer_breakdown(
    reference: str,
    hypothesis: str,
) -> dict:
    """Compute WER with insertions, deletions, substitutions breakdown.

    Returns:
        Dict with 'wer', 'insertions', 'deletions', 'substitutions', 'n_ref_words'.
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return {"wer": 0.0 if m == 0 else float(m),
                "insertions": m, "deletions": 0, "substitutions": 0, "n_ref_words": 0}

    # DP with backtracking
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    ops = np.zeros((n + 1, m + 1), dtype=np.int32)  # 0=match, 1=sub, 2=del, 3=ins
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 0
            else:
                choices = [dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]
                best = np.argmin(choices)
                dp[i][j] = 1 + choices[best]
                ops[i][j] = best + 1  # 1=sub, 2=del, 3=ins

    # Backtrace to count S, D, I
    i, j = n, m
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] == 0:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and ops[i][j] == 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and (j == 0 or ops[i][j] == 2):
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1

    return {
        "wer": float(dp[n][m]) / float(n),
        "insertions": ins,
        "deletions": dels,
        "substitutions": subs,
        "n_ref_words": n,
    }


def character_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    """Character Error Rate.

    Same as WER but at character level. Useful for comparing
    fine-grained ASR accuracy.

    Returns:
        CER as a float.
    """
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return 0.0 if m == 0 else float(m)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return float(dp[n][m]) / float(n)


def output_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    """Compute output SNR in dB.

    SNR = 10 * log10(P_signal / P_noise)

    Args:
        signal: Clean or target signal component.
        noise: Noise or residual component.

    Returns:
        SNR in dB.
    """
    p_signal = float(np.mean(signal.astype(np.float64) ** 2))
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))
    if p_noise < 1e-10:
        return float("inf")
    return 10.0 * np.log10(p_signal / p_noise)


# ---------------------------------------------------------------------------
# Diarization metrics
# ---------------------------------------------------------------------------

def diarization_error_rate(
    reference_segments: List[Tuple[float, float, str]],
    hypothesis_segments: List[Tuple[float, float, str]],
) -> float:
    """Simplified Diarization Error Rate.

    DER = (FA + MISS + CONFUSION) / TOTAL_REFERENCE_DURATION

    For full DER, use pyannote.metrics. This is a simplified version
    that computes overlap-based error.

    Args:
        reference_segments: List of (start, end, speaker_id) ground truth.
        hypothesis_segments: List of (start, end, speaker_id) hypothesis.

    Returns:
        DER as a float.
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        ref_ann = Annotation()
        for start, end, spk in reference_segments:
            ref_ann[Segment(start, end)] = spk

        hyp_ann = Annotation()
        for start, end, spk in hypothesis_segments:
            hyp_ann[Segment(start, end)] = spk

        metric = DiarizationErrorRate()
        return float(metric(ref_ann, hyp_ann))

    except ImportError:
        logger.warning(
            "pyannote.metrics not installed. Using simplified DER calculation."
        )
        # Simplified: compute total reference duration
        total_ref = sum(end - start for start, end, _ in reference_segments)
        if total_ref == 0:
            return 0.0

        # This is a placeholder; real DER needs pyannote.metrics
        return float("nan")


# ---------------------------------------------------------------------------
# VAD metrics (VAD_IMPLEMENTATION.md Section 8)
# ---------------------------------------------------------------------------

def compute_vad_metrics(
    predicted: List[bool],
    reference: List[bool],
) -> Dict[str, float]:
    """Compute VAD accuracy metrics.

    Args:
        predicted: Per-frame VAD decisions (True = speech).
        reference: Per-frame ground truth (True = speech).

    Returns:
        Dict with precision, recall, f1, f2, accuracy, tp, fp, fn, tn.
    """
    min_len = min(len(predicted), len(reference))
    pred = predicted[:min_len]
    ref = reference[:min_len]

    tp = sum(1 for p, r in zip(pred, ref) if p and r)
    fp = sum(1 for p, r in zip(pred, ref) if p and not r)
    fn = sum(1 for p, r in zip(pred, ref) if not p and r)
    tn = sum(1 for p, r in zip(pred, ref) if not p and not r)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(min_len, 1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_phi_nn_contamination(
    vad_noise_labels: List[bool],
    true_speech_labels: List[bool],
) -> float:
    """Compute Phi_nn contamination rate.

    Measures the fraction of frames marked as noise by VAD
    that were actually speech. High contamination degrades MVDR.

    Target: < 5%

    Args:
        vad_noise_labels: Per-frame noise decisions from VAD.
        true_speech_labels: Per-frame ground truth speech labels.

    Returns:
        Contamination rate [0, 1].
    """
    min_len = min(len(vad_noise_labels), len(true_speech_labels))
    noise_frames = sum(1 for i in range(min_len) if vad_noise_labels[i])
    contaminated = sum(
        1 for i in range(min_len)
        if vad_noise_labels[i] and true_speech_labels[i]
    )

    if noise_frames == 0:
        return 0.0
    return contaminated / noise_frames


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

def real_time_factor(
    processing_time_s: float,
    audio_duration_s: float,
) -> float:
    """Real-Time Factor: processing time / audio duration.

    RTF < 1.0 means real-time capable.

    Args:
        processing_time_s: Time to process in seconds.
        audio_duration_s: Duration of input audio in seconds.

    Returns:
        RTF (dimensionless).
    """
    if audio_duration_s <= 0:
        return float("inf")
    return processing_time_s / audio_duration_s
