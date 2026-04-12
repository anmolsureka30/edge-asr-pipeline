"""
CascadePipeline: Sequential pipeline with no feedback.

Level 1 architecture (CLAUDE.md Section 6):
    SSL -> Beamforming -> Enhancement -> [Separation] -> ASR -> [Diarization]

Each module is independent. Data flows forward only.
This is the baseline against which Level 2 (guided) is compared.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..modules.base import BaseModule

logger = logging.getLogger(__name__)


class CascadePipeline:
    """Sequential pipeline executing modules in order.

    Modules are executed in the order they are added. Each module
    receives the accumulated data dictionary and adds its outputs.

    Usage:
        pipeline = CascadePipeline()
        pipeline.add_module(ssl)
        pipeline.add_module(beamformer)
        pipeline.add_module(enhancer)
        pipeline.add_module(asr)

        data = scene.to_pipeline_dict()
        result = pipeline.run(data)
    """

    def __init__(self, name: str = "cascade"):
        self.name = name
        self.modules: List[BaseModule] = []
        self._logger = logging.getLogger(f"{__name__}.{name}")

    def add_module(self, module: BaseModule) -> "CascadePipeline":
        """Add a module to the pipeline.

        Args:
            module: A BaseModule instance.

        Returns:
            self (for chaining).
        """
        self.modules.append(module)
        self._logger.info(f"Added module: {module.name}")
        return self

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full pipeline on input data.

        Args:
            data: Input data dictionary (from AcousticScene.to_pipeline_dict()).

        Returns:
            Data dictionary with all module outputs accumulated.
        """
        self._logger.info(
            f"Running pipeline '{self.name}' with {len(self.modules)} modules"
        )

        total_start = time.perf_counter()
        module_timings = []

        for module in self.modules:
            module_start = time.perf_counter()
            try:
                data = module.process(data)
                elapsed = (time.perf_counter() - module_start) * 1000.0
                module_timings.append((module.name, elapsed))
                self._logger.info(f"  {module.name}: {elapsed:.1f}ms")
            except Exception as e:
                self._logger.error(f"  {module.name} FAILED: {e}")
                data[f"{module.name}_error"] = str(e)
                raise

        total_elapsed = (time.perf_counter() - total_start) * 1000.0

        # Store pipeline metadata
        data["pipeline_name"] = self.name
        data["pipeline_modules"] = [m.name for m in self.modules]
        data["pipeline_timings"] = module_timings
        data["pipeline_total_ms"] = total_elapsed

        self._logger.info(f"Pipeline complete: {total_elapsed:.1f}ms total")
        return data

    def get_config(self) -> Dict[str, Any]:
        """Return full pipeline configuration."""
        return {
            "pipeline_name": self.name,
            "pipeline_type": "cascade",
            "modules": [
                {"name": m.name, "config": m.get_config()}
                for m in self.modules
            ],
        }

    def __repr__(self) -> str:
        module_names = " -> ".join(m.name for m in self.modules)
        return f"CascadePipeline({module_names})"
