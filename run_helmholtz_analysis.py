#!/usr/bin/env python
"""
Executable script for single galaxy Helmholtz Vector Field Decomposition analysis.
Decomposes 3D velocity field into Solenoidal (curl-only) and Compressive (div-only) modes.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import hydro_spectral as hs

def parse_args():
    parser = argparse.ArgumentParser(description="Helmholtz Velocity Field Decomposition")
    parser.add_argument("--basePath", type=str, default=hs.Config.basePath)
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--subhaloID", type=int, default=0)
    parser.add_argument("--Ng", type=int, default=32)
    parser.add_argument("--Rmax", type=float, default=200.0)
    parser.add_argument("--outdir", type=str, default="spectrum_plots")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.dry_run:
        print("=== RUNNING DRY RUN FOR HELMHOLTZ DECOMPOSITION ===")
        np.random.seed(42)
        N = 8000
        # Create a synthetic rotating disk field (mostly solenoidal)
        pos = np.random.uniform(-100, 100, (N, 3))
        r = np.linalg.norm(pos[:, :2], axis=1) + 1e-5
        v_theta = 200.0 * (r / 100.0) * np.exp(-r / 50.0)
        vx = -v_theta * (pos[:, 1] / r)
        vy = v_theta * (pos[:, 0] / r)
        vz = np.random.normal(0, 20, N)
        vel = np.column_stack([vx, vy, vz])
        center = np.zeros(3)
        
        vgrid = hs.construct_velocity_grid(pos, vel, center, args.Rmax, Ng=args.Ng)
        helm = hs.helmholtz_decomposition(vgrid, Ng=args.Ng)
        
        print("\nHelmholtz Energy Decomposition Results:")
        print(f"  Solenoidal Energy (E_sol) : {helm['E_sol']:.2e} ({helm['solenoidal_ratio']*100:.1f}%)")
        print(f"  Compressive Energy (E_comp): {helm['E_comp']:.2e} ({helm['compressive_ratio']*100:.1f}%)")
        
        # Plot bar chart
        plt.figure(figsize=(6, 5))
        plt.bar(["Solenoidal ($v_{sol}$)", "Compressive ($v_{comp}$)"], [helm['E_sol'], helm['E_comp']], color=["navy", "crimson"])
        plt.ylabel("Kinetic Energy Density")
        plt.title("Synthetic Rotating Disk: Helmholtz Energy Split")
        out_bar = os.path.join(args.outdir, "helmholtz_energy_bar.png")
        plt.tight_layout()
        plt.savefig(out_bar, dpi=150)
        plt.close()
        print(f"Saved plot: {out_bar}")
        return

    print("Use --dry-run for synthetic demonstration.")

if __name__ == "__main__":
    main()
