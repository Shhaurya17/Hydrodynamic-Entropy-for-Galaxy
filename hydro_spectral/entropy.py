"""
Hydrodynamic Spectral Entropy calculator module.
"""

import numpy as np
from .grid import construct_velocity_grid
from .spectrum import compute_energy_spectrum, low_k_fraction

def spectral_entropy(Ek: np.ndarray) -> float:
    """
    Compute normalized hydrodynamic spectral entropy S_H = - sum(p_i log(p_i)) / log(N).
    
    Parameters
    ----------
    Ek : np.ndarray
        1D kinetic energy spectrum.
        
    Returns
    -------
    float
        Normalized spectral entropy value between 0.0 (concentrated power) and 1.0 (equipartition).
    """
    if Ek is None or len(Ek) <= 1:
        return 0.0
        
    Ek_non_zero = Ek[Ek > 0]
    if len(Ek_non_zero) <= 1:
        return 0.0
        
    P = Ek_non_zero / np.sum(Ek_non_zero)
    P = P[P > 0]
    
    S = -np.sum(P * np.log(P))
    S_max = np.log(len(P))
    
    if S_max == 0:
        return 0.0
        
    return float(S / S_max)

def spectral_entropy_numpy(Ek: np.ndarray) -> float:
    return spectral_entropy(Ek)

def compute_hydro_entropy(pos: np.ndarray, vel: np.ndarray, center: np.ndarray, Rmax: float, Ng: int = 32, min_cells: int = 100):
    """
    Convenience function to compute velocity grid, spectrum, and spectral entropy + low_k_fraction.
    """
    grid = construct_velocity_grid(pos, vel, center, Rmax, Ng=Ng, min_cells=min_cells)
    if grid is None:
        return None, None
    Ek = compute_energy_spectrum(grid, Ng=Ng)
    return spectral_entropy(Ek), low_k_fraction(Ek)

def entropy_at_radius(pos_all: np.ndarray, vel_all: np.ndarray, center: np.ndarray, Rmax: float, Ng: int = 32, min_cells: int = 100):
    """
    Compute hydrodynamic spectral entropy for particles inside spherical aperture Rmax.
    """
    r_all = np.linalg.norm(pos_all - center, axis=1)
    mask = r_all < Rmax
    if np.sum(mask) < min_cells:
        return None
        
    grid = construct_velocity_grid(pos_all[mask], vel_all[mask], center, Rmax, Ng=Ng, min_cells=min_cells)
    if grid is None:
        return None
        
    Ek = compute_energy_spectrum(grid, Ng=Ng)
    return spectral_entropy(Ek), low_k_fraction(Ek)
