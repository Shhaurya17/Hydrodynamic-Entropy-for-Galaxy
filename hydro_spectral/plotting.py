"""
Visualization and diagnostic plotting module for spectra, fits, and entropy evolution.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

def save_combined_spectrum_subplots(spectra_data: list, component: str, out_dir: str = "spectrum_plots", ncols: int = 4) -> str:
    """
    Save combined subplots of raw spectra E(k) across snapshots into a grid image.
    """
    if not spectra_data:
        return ""
        
    os.makedirs(out_dir, exist_ok=True)
    N = len(spectra_data)
    nrows = math.ceil(N / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = np.array([axes])
        
    for i, item in enumerate(spectra_data):
        ax = axes_flat[i]
        snap = item["Snapshot"]
        subid = item["SubhaloID"]
        K = item["k"]
        Ek = item["Ek"]
        
        valid = (K > 0) & (Ek > 0)
        ax.loglog(K[valid], Ek[valid], 'o-', ms=3, label=f"Snap {snap}")
        ax.set_title(f"Snap {snap} (SubID {subid})", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        
    # Hide unused axes
    for j in range(N, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    fig.text(0.5, 0.02, "Wavenumber $k$", ha="center", fontsize=12)
    fig.text(0.02, 0.5, f"{component.upper()} Energy Spectrum $E(k)$", va="center", rotation="vertical", fontsize=12)
    
    out_path = os.path.join(out_dir, f"combined_spectra_{component}.png")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path

def save_combined_slope_fit_subplots(spectra_data: list, component: str, out_dir: str = "spectrum_plots", ncols: int = 4) -> str:
    """
    Save combined subplots showing power-law linear fit region across snapshots.
    """
    from .powerlaw import find_best_linear_region_loglog
    
    if not spectra_data:
        return ""
        
    os.makedirs(out_dir, exist_ok=True)
    N = len(spectra_data)
    nrows = math.ceil(N / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = np.array([axes])
        
    for i, item in enumerate(spectra_data):
        ax = axes_flat[i]
        snap = item["Snapshot"]
        K = item["k"]
        Ek = item["Ek"]
        
        fit = find_best_linear_region_loglog(K, Ek)
        valid = (K > 0) & (Ek > 0)
        ax.loglog(K[valid], Ek[valid], 'o', color="gray", alpha=0.5, ms=3)
        
        if len(fit["k_fit"]) > 0:
            ax.loglog(fit["k_fit"], fit["Ek_fit_line"], 'r-', lw=2, label=f"$\\alpha={fit['alpha']:.2f}$\n$R^2={fit['R2']:.2f}$")
            
        ax.set_title(f"Snap {snap}", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        
    for j in range(N, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    out_path = os.path.join(out_dir, f"combined_slope_fits_{component}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path

# Backward compatibility alias
save_powerlaw_subplots = save_combined_slope_fit_subplots

def plot_entropy_evolution(df_gas, df_dm=None, out_path="entropy_evolution.png"):
    """
    Plot spectral entropy evolution across snapshot numbers for Gas and DM.
    """
    plt.figure(figsize=(8, 5))
    if df_gas is not None and "Entropy" in df_gas:
        plt.plot(df_gas["Snapshot"], df_gas["Entropy"], "o-", label="Gas", lw=2)
    if df_dm is not None and "Entropy" in df_dm:
        plt.plot(df_dm["Snapshot"], df_dm["Entropy"], "s-", label="Dark Matter", lw=2)
        
    plt.xlabel("Snapshot Number")
    plt.ylabel("Hydrodynamic Entropy $S_H$")
    plt.title("Spectral Entropy Evolution along Main Progenitor Branch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
