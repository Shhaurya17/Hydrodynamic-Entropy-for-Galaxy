#!/usr/bin/env python
"""
Executable script for Hydrodynamic Spectral Entropy Analysis Pipeline.
Replaces the main analysis notebook (spectral_entropy_v5.ipynb).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hydro_spectral as hs

def parse_args():
    parser = argparse.ArgumentParser(description="Hydrodynamic Spectral Entropy Pipeline")
    parser.add_argument("--basePath", type=str, default=hs.Config.basePath, help="IllustrisTNG output directory path")
    parser.add_argument("--snap0", type=int, default=99, help="Target redshift z=0 snapshot number")
    parser.add_argument("--target_subhalo", type=int, default=0, help="Target subhalo ID at z=0")
    parser.add_argument("--stride", type=int, default=5, help="Snapshot stride along Main Progenitor Branch")
    parser.add_argument("--Ng", type=int, default=32, help="3D grid resolution")
    parser.add_argument("--Rmax", type=float, default=200.0, help="Spherical aperture radius (ckpc/h)")
    parser.add_argument("--outdir", type=str, default="spectrum_plots", help="Output directory for plots")
    parser.add_argument("--dry-run", action="store_true", help="Run in synthetic test mode without loading TNG files")
    return parser.parse_args()

def run_dry_run(args):
    print("=== RUNNING DRY RUN WITH SYNTHETIC DATA ===")
    np.random.seed(42)
    N_part = 5000
    pos = np.random.uniform(-100, 100, (N_part, 3))
    vel = np.random.normal(0, 200, (N_part, 3))
    center = np.zeros(3)
    
    prof = hs.profile_cpu(pos, vel, center, args.Rmax, Ng=args.Ng)
    print("\n" + hs.format_markdown_report(prof))
    
    # Test Helmholtz decomposition
    grid = hs.construct_velocity_grid(pos, vel, center, args.Rmax, Ng=args.Ng)
    helm = hs.helmholtz_decomposition(grid, Ng=args.Ng)
    print(f"Solenoidal Energy Ratio: {helm['solenoidal_ratio']:.4f}")
    print(f"Compressive Energy Ratio: {helm['compressive_ratio']:.4f}")
    
    # Test Power Law Fit
    Ek = hs.compute_energy_spectrum(grid, Ng=args.Ng)
    K = hs.get_k_axis(len(Ek))
    fit = hs.find_best_linear_region_loglog(K, Ek)
    print(f"Power-Law Fit Alpha: {fit['alpha']:.4f}, R2: {fit['R2']:.4f}")
    
    print("\nDry run completed successfully.")

def main():
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading MPB tree for Subhalo {args.target_subhalo} starting from Snapshot {args.snap0}...")
    
    try:
        import illustris_python as il
    except ImportError:
        print("Error: illustris_python library is required for real TNG dataset processing.")
        print("Run with --dry-run to test code on synthetic particle fields.")
        sys.exit(1)
        
    snaps, subhalos = hs.load_mpb_branch(args.basePath, snap0=args.snap0, target_subhalo=args.target_subhalo, snap_stride=args.stride)
    if not snaps:
        print("No progenitor tree found. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(snaps)} snapshots along MPB.")
    results_gas = []
    gas_spectra_data = []
    
    for snap, subID in zip(snaps, subhalos):
        data = hs.load_subhalo_data(args.basePath, snap, subID, component="gas")
        if data is None:
            continue
            
        prof = hs.profile_cpu(data["pos"], data["vel"], data["center"], args.Rmax, Ng=args.Ng)
        if prof is None:
            continue
            
        results_gas.append({
            "Snapshot": snap,
            "SubhaloID": subID,
            "Entropy": prof["Entropy"],
            "LowKFrac": prof["LowKFrac"],
            "GFLOPs": prof["GFLOPs"]
        })
        gas_spectra_data.append({
            "Snapshot": snap,
            "SubhaloID": subID,
            "k": hs.get_k_axis(len(prof["Ek"])),
            "Ek": prof["Ek"]
        })
        
    df_gas = pd.DataFrame(results_gas)
    if not df_gas.empty:
        df_gas = df_gas.sort_values("Snapshot")
        print("\n=== GAS SPECTRAL ENTROPY RESULTS ===")
        print(df_gas.to_string(index=False))
        
        gas_plot_path = hs.plot_entropy_evolution(df_gas, out_path=os.path.join(args.outdir, "gas_entropy_evolution.png"))
        print(f"Saved plot: {gas_plot_path}")

if __name__ == "__main__":
    main()
