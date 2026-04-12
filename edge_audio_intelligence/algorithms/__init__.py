"""
Core mathematical algorithms — separated from simulation and pipeline code.

Each file contains a self-contained algorithm implementation with:
- Full mathematical description in the docstring
- No dependencies on simulation/dashboard/pipeline code
- Pure functions or stateless classes operating on numpy arrays

These can be imported and used anywhere: in the pipeline modules,
in the testbench, in experiments, or independently.
"""

from .rir import RIRGenerator, image_source_rir, convolve_rir, sabine_rt60
from .signal_mixing import mix_sources_through_rirs, add_noise_at_snr
from .doa import compute_doa_azimuth, compute_doa_azimuth_elevation
