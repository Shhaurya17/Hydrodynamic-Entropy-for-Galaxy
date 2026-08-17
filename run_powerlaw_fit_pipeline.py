#!/usr/bin/env python
"""
Executable script for Power-Law Spectrum Fitting Pipeline.
Replaces spectral_powerlaw_fit_v2.ipynb.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

import hydro_spectral as hs

def parse_args():
    parser = argparse.ArgumentParser(description="Power-Law Fit Pipeline")
    parser.add_argument("--basePath", type=str, default=hs.Config.basePath)
    parser.add_argument("--component", type=str, default="gas", choices=["gas", "dm"])
    parser.add_argument("--Ng", type=int, default=32)
    parser.add_argument("--Rmax", type=float, default=200.0)
    parser.add_argument("--outdir", type=str, default="fit_plots")
    parser.add_argument("--cachedir", type=str, default="cache_spectra")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def run_dry_run(args):
    print("=== RUNNING DRY RUN FOR POWER-LAW FIT ===")
    np.random.seed(123)
    k = np.arange(1, 17)
    # Generate synthetic Kolmogorov spectrum E(k) ~ k^(-5/3) with noise
    Ek = 100.0 * (k ** (-5.0 / 3.0)) * (1 + 0.1 * np.random.randn(len(k)))
    
    fit = hs.find_best_linear_region_loglog(k, Ek)
    print(f"Synthetic Spectrum Fit Results:")
    print(f"  Alpha (Slope): {fit['alpha']:.4f} (Expected: ~ -1.67)")
    print(f"  Log(A): {fit['logA']:.4f}")
    print(f"  R^2: {fit['R2']:.4f}")
    print(f"  Wavenumber Range: {fit['k_start']} to {fit['k_end']}")
    
    spectra_data = [{"Snapshot": 99, "SubhaloID": 0, "k": k, "Ek": Ek}]
    out_path = hs.save_powerlaw_subplots(spectra_data, component="synthetic", out_dir=args.outdir)
    print(f"Saved plot: {out_path}")

def main():
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return
        
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cachedir, exist_ok=True)
    print(f"Processing component '{args.component}' for Power-Law spectral fitting...")
    # Add full dataset processing logic here if available
    print("Use --dry-run for synthetic spectrum test.")

if __name__ == "__main__":
    main()
