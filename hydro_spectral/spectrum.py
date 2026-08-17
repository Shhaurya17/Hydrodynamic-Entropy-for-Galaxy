"""
3D FFT Kinetic Energy Spectrum computation and k-binning module.
"""

import numpy as np

def compute_energy_spectrum(vgrid: np.ndarray, Ng: int = None) -> np.ndarray:
    """
    Compute 1D isotropic kinetic energy spectrum E(k) from a 3D velocity grid vgrid(Ng, Ng, Ng, 3).
    
    Parameters
    ----------
    vgrid : np.ndarray
        3D velocity grid array of shape (Ng, Ng, Ng, 3).
    Ng : int, optional
        Grid resolution along spatial dimension. Inferred from vgrid if None.
        
    Returns
    -------
    Ek : np.ndarray
        1D kinetic energy spectrum indexed by k = 0, 1, ..., k_max.
    """
    if Ng is None:
        Ng = vgrid.shape[0]
        
    vk_x = np.fft.fftn(vgrid[:, :, :, 0])
    vk_y = np.fft.fftn(vgrid[:, :, :, 1])
    vk_z = np.fft.fftn(vgrid[:, :, :, 2])
    
    P_k = 0.5 * (np.abs(vk_x)**2 + np.abs(vk_y)**2 + np.abs(vk_z)**2) / (Ng**3)
    
    kx = np.fft.fftfreq(Ng) * Ng
    ky = np.fft.fftfreq(Ng) * Ng
    kz = np.fft.fftfreq(Ng) * Ng
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    k_int = np.round(K).astype(int)
    max_k = int(np.max(k_int))
    
    Ek = np.bincount(k_int.ravel(), weights=P_k.ravel(), minlength=max_k + 1)
    return Ek

def get_k_axis(nbins: int) -> np.ndarray:
    """
    Return wavenumber grid integer indices k = 0, 1, ..., nbins - 1.
    """
    return np.arange(nbins)

def compute_energy_spectrum_cpu(vgrid: np.ndarray, ng: int = 32) -> np.ndarray:
    """
    Alias/wrapper for compute_energy_spectrum.
    """
    return compute_energy_spectrum(vgrid, Ng=ng)

def low_k_fraction(Ek: np.ndarray, frac: float = 0.1) -> float:
    """
    Calculate the fraction of total kinetic energy contained in the lowest k modes (largest scales).
    
    Parameters
    ----------
    Ek : np.ndarray
        1D kinetic energy spectrum.
    frac : float
        Fraction of k-bins to include from the start (default: 0.1 = 10%).
        
    Returns
    -------
    float
        Ratio of low-k kinetic energy to total kinetic energy.
    """
    if Ek is None or len(Ek) == 0:
        return 0.0
    tot = np.sum(Ek)
    if tot == 0:
        return 0.0
    n = max(1, int(len(Ek) * frac))
    return float(np.sum(Ek[:n]) / tot)
