"""
Testbench: Acoustic simulation and evaluation framework.

Generates controlled acoustic scenes using pyroomacoustics,
mixes signals with noise, and evaluates pipeline modules.
See ACOUSTIC_LAB.md for design philosophy.
"""

from .scene import AcousticScene, SceneConfig, MicArrayConfig
from .simulator import RIRSimulator
from .evaluator import PipelineEvaluator
from .visualizer import plot_scene_layout, plot_pipeline_signals
