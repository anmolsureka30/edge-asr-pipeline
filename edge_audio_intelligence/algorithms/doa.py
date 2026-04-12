"""
Direction of Arrival (DOA) computation.

Given a microphone array center and a source position, compute
the azimuth (and optionally elevation) angle.

Azimuth: angle in the horizontal plane from positive x-axis,
measured counter-clockwise. Range [0, 360).

Elevation: angle from horizontal plane. Range [-90, 90].
"""

import numpy as np
from typing import List


def compute_doa_azimuth(
    mic_center: np.ndarray,
    source_pos: np.ndarray,
) -> float:
    """Compute Direction of Arrival (azimuth only) in degrees.

    azimuth = atan2(dy, dx)  in degrees, wrapped to [0, 360)

    Args:
        mic_center: [x, y, z] microphone array center.
        source_pos: [x, y, z] source position.

    Returns:
        Azimuth in degrees [0, 360).
    """
    dx = source_pos[0] - mic_center[0]
    dy = source_pos[1] - mic_center[1]
    azimuth = np.degrees(np.arctan2(dy, dx)) % 360.0
    return float(azimuth)


def compute_doa_azimuth_elevation(
    mic_center: np.ndarray,
    source_pos: np.ndarray,
) -> tuple:
    """Compute Direction of Arrival (azimuth + elevation) in degrees.

    azimuth   = atan2(dy, dx)
    elevation = atan2(dz, sqrt(dx^2 + dy^2))

    Args:
        mic_center: [x, y, z] microphone array center.
        source_pos: [x, y, z] source position.

    Returns:
        Tuple of (azimuth_deg, elevation_deg).
    """
    dx = source_pos[0] - mic_center[0]
    dy = source_pos[1] - mic_center[1]
    dz = source_pos[2] - mic_center[2]

    azimuth = np.degrees(np.arctan2(dy, dx)) % 360.0
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    elevation = np.degrees(np.arctan2(dz, horizontal_dist))

    return float(azimuth), float(elevation)


def compute_tdoa(
    mic1_pos: np.ndarray,
    mic2_pos: np.ndarray,
    source_pos: np.ndarray,
    speed_of_sound: float = 343.0,
) -> float:
    """Compute Time Difference of Arrival between two microphones.

    TDOA = (d1 - d2) / c

    where d1, d2 are distances from source to mic1, mic2.

    Args:
        mic1_pos: [x, y, z] first microphone position.
        mic2_pos: [x, y, z] second microphone position.
        source_pos: [x, y, z] source position.
        speed_of_sound: Speed of sound in m/s.

    Returns:
        TDOA in seconds. Positive means sound arrives at mic1 first.
    """
    d1 = np.linalg.norm(np.array(source_pos) - np.array(mic1_pos))
    d2 = np.linalg.norm(np.array(source_pos) - np.array(mic2_pos))
    return float((d1 - d2) / speed_of_sound)
