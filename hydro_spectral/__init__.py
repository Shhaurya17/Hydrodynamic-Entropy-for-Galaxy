"""
HydroSpectral: Modular library for hydrodynamic spectral entropy, FFT power spectra,
Helmholtz decomposition, and power-law fitting of cosmological simulation velocity fields.
"""

__version__ = "1.0.0"

from .config import Config
from .grid import construct_velocity_grid, filter_aperture
from .spectrum import compute_energy_spectrum, get_k_axis, low_k_fraction
from .entropy import spectral_entropy, compute_hydro_entropy, entropy_at_radius
from .helmholtz import helmholtz_decomposition, power
from .thermo import gas_temperature, thermodynamic_entropy_per_particle, compute_thermo_entropy
from .powerlaw import find_best_linear_region_loglog, _r2_score
from .data_loader import load_subhalo_data, load_mpb_branch
from .profiling import profile_cpu, format_markdown_report
from .caching import spectra_list_to_df, spectra_df_to_list, cache_tag
from .plotting import (
    save_combined_spectrum_subplots,
    save_combined_slope_fit_subplots,
    save_powerlaw_subplots,
    plot_entropy_evolution
)

__all__ = [
    "Config",
    "construct_velocity_grid",
    "filter_aperture",
    "compute_energy_spectrum",
    "get_k_axis",
    "low_k_fraction",
    "spectral_entropy",
    "compute_hydro_entropy",
    "entropy_at_radius",
    "helmholtz_decomposition",
    "power",
    "gas_temperature",
    "thermodynamic_entropy_per_particle",
    "compute_thermo_entropy",
    "find_best_linear_region_loglog",
    "_r2_score",
    "load_subhalo_data",
    "load_mpb_branch",
    "profile_cpu",
    "format_markdown_report",
    "spectra_list_to_df",
    "spectra_df_to_list",
    "cache_tag",
    "save_combined_spectrum_subplots",
    "save_combined_slope_fit_subplots",
    "save_powerlaw_subplots",
    "plot_entropy_evolution",
]
