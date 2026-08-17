"""
Configuration parameters for HydroSpectral analysis pipeline.
"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    basePath: str = os.getenv("TNG_BASE_PATH", "/home/jovyan/Simulations/TNG50-1/output")
    TARGET_SUBHALO: int = 0
    Ng: int = 32
    Rmax: float = 200.0
    h: float = 0.6774
    MIN_CELLS: int = 100
    
    SPECTRUM_OUTPUT_DIR: str = "spectrum_plots"
    FIT_OUTPUT_DIR: str = "fit_plots"
    CACHE_DIR: str = "cache_spectra"
    CACHE_VERSION: str = "v2"
    
    SNAP_STRIDE: int = 5
    MIN_LINEAR_POINTS: int = 4
    MAX_LINEAR_POINTS: int = 15
    FORCE_RECOMPUTE_SPECTRA: bool = False
    FORCE_REFIT: bool = False

default_config = Config()
