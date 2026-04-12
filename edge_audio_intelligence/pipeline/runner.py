"""
ExperimentRunner: Orchestrates experiments with logging and reproducibility.

Handles the experiment workflow (CLAUDE.md Section 4.1):
1. Generate or load acoustic scenes
2. Run the pipeline
3. Compute all metrics at every probe point
4. Save results
5. Append summary to RESEARCH_LOG.md
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..testbench.scene import AcousticScene, SceneConfig
from ..testbench.simulator import RIRSimulator
from ..testbench.evaluator import PipelineEvaluator, PipelineResult
from .cascade import CascadePipeline

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs experiments with full logging, evaluation, and reproducibility.

    Usage:
        runner = ExperimentRunner(
            name="01_baseline",
            pipeline=pipeline,
            results_dir="results/",
        )
        runner.run_on_scenes(scenes)
        runner.save_results()
        runner.print_summary()
    """

    def __init__(
        self,
        name: str,
        pipeline: CascadePipeline,
        results_dir: str = "results",
        seed: int = 42,
    ):
        self.name = name
        self.pipeline = pipeline
        self.results_dir = Path(results_dir)
        self.seed = seed
        self.simulator = RIRSimulator(seed=seed)
        self.evaluator = PipelineEvaluator(results_dir=results_dir)

        self.results: List[PipelineResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def run_on_scenes(
        self,
        scenes: List[AcousticScene],
    ) -> List[PipelineResult]:
        """Run the pipeline on a list of pre-generated scenes.

        Args:
            scenes: List of AcousticScene objects.

        Returns:
            List of PipelineResult objects.
        """
        self.start_time = datetime.now()
        self.results = []

        logger.info(
            f"Experiment '{self.name}': running on {len(scenes)} scenes "
            f"with pipeline: {self.pipeline}"
        )

        for i, scene in enumerate(scenes):
            scene_id = f"{self.name}_scene_{i:03d}"
            logger.info(
                f"  Scene {i+1}/{len(scenes)}: "
                f"SNR={scene.config.snr_db}dB, RT60={scene.config.rt60}s"
            )

            # Convert scene to pipeline dict
            data = scene.to_pipeline_dict()

            # Run pipeline
            try:
                data = self.pipeline.run(data)
            except Exception as e:
                logger.error(f"  Pipeline failed on {scene_id}: {e}")
                continue

            # Evaluate
            result = self.evaluator.run_full_evaluation(data, scene, scene_id)
            self.results.append(result)

        self.end_time = datetime.now()
        logger.info(
            f"Experiment '{self.name}' complete: "
            f"{len(self.results)}/{len(scenes)} scenes processed"
        )
        return self.results

    def run_on_config(
        self,
        base_config: SceneConfig,
        mode: str = "corner_cases",
    ) -> List[PipelineResult]:
        """Generate scenes from config and run the pipeline.

        Args:
            base_config: Base scene configuration.
            mode: 'corner_cases' (4 scenes), 'full_grid' (18 scenes),
                  or 'quick' (1 easy scene).

        Returns:
            List of PipelineResult objects.
        """
        if mode == "corner_cases":
            scenes = self.simulator.generate_corner_cases(base_config)
        elif mode == "full_grid":
            scenes = self.simulator.generate_grid_scenes(base_config)
        elif mode == "quick":
            import dataclasses
            easy_config = dataclasses.replace(base_config, snr_db=30.0, rt60=0.0)
            scenes = [self.simulator.generate_scene(easy_config)]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return self.run_on_scenes(scenes)

    def save_results(self) -> Path:
        """Save all results to JSON."""
        path = self.evaluator.save_results(f"{self.name}_results.json")

        # Also save experiment config
        config_path = self.results_dir / "logs" / f"{self.name}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "experiment_name": self.name,
            "pipeline_config": self.pipeline.get_config(),
            "seed": self.seed,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "n_scenes": len(self.results),
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Saved config to {config_path}")
        return path

    def print_summary(self) -> str:
        """Print and return summary table."""
        return self.evaluator.print_summary_table()

    def append_to_research_log(
        self,
        log_path: str,
        hypothesis: str = "",
        conclusion: str = "",
    ) -> None:
        """Append experiment summary to RESEARCH_LOG.md.

        Args:
            log_path: Path to RESEARCH_LOG.md.
            hypothesis: The experiment's hypothesis.
            conclusion: Post-experiment conclusion.
        """
        log_path = Path(log_path)

        entry = f"\n\n## Experiment: {self.name}\n"
        entry += f"**Date:** {self.start_time.strftime('%Y-%m-%d %H:%M') if self.start_time else 'N/A'}\n\n"

        if hypothesis:
            entry += f"**Hypothesis:** {hypothesis}\n\n"

        entry += f"**Pipeline:** {self.pipeline}\n\n"

        # Summary table
        entry += "**Results:**\n```\n"
        entry += self.evaluator.print_summary_table()
        entry += "\n```\n\n"

        if conclusion:
            entry += f"**Conclusion:** {conclusion}\n\n"

        entry += "---\n"

        with open(log_path, "a") as f:
            f.write(entry)

        logger.info(f"Appended to {log_path}")
