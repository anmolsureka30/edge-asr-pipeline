"""
BaseModule: Abstract base class for all pipeline modules.

Every module in the pipeline implements this interface, enabling plug-and-play
swapping and standardized benchmarking. See CLAUDE.md Section 3.2.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """Abstract base class for all pipeline modules.

    All pipeline stages (SSL, beamforming, enhancement, separation, ASR,
    diarization) must subclass this and implement process() and get_config().

    Data flow between modules uses standardized dictionaries.
    See CLAUDE.md Section 3.3 for the complete key specification.
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        self.name = name
        self.sample_rate = sample_rate
        self._logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results.

        Args:
            data: Dictionary with standardized keys per module type.

        Returns:
            The input dictionary with additional keys added by this module.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return all configuration parameters for experiment logging."""
        pass

    def measure_latency(self, data: Dict[str, Any], n_runs: int = 100) -> float:
        """Measure average processing latency in milliseconds.

        Args:
            data: Input data dictionary.
            n_runs: Number of runs to average over.

        Returns:
            Average latency in milliseconds.
        """
        # Warm-up run
        self.process(data.copy())

        times = []
        for _ in range(n_runs):
            data_copy = data.copy()
            t0 = time.perf_counter()
            self.process(data_copy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        avg_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        self._logger.info(
            f"{self.name} latency: {avg_ms:.2f} +/- {std_ms:.2f} ms "
            f"(n={n_runs})"
        )
        return avg_ms

    def count_parameters(self) -> int:
        """Return total trainable parameters (0 for non-neural methods)."""
        return 0

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for one forward pass.

        Subclasses should override with algorithm-specific estimates.
        FLOPs = 2 * MACs for most operations.
        """
        return self.estimate_macs(data) * 2

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Estimate MACs (Multiply-Accumulate Operations) for one forward pass.

        1 MAC = 1 multiplication + 1 addition = 2 FLOPs.
        MACs is the standard compute cost metric for edge deployment.

        Subclasses should override with algorithm-specific estimates.
        See utils/profiling.py for estimation helpers:
            estimate_macs_for_fft()
            estimate_macs_for_convolution()
            estimate_macs_for_dwt()
            estimate_macs_for_whisper()
        """
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
