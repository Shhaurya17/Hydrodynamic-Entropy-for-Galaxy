"""
Thermodynamic gas entropy and temperature computation module for IllustrisTNG gas cells.
"""

import numpy as np

# Physical Constants (CGS)
KB = 1.380649e-16       # erg/K
MP = 1.6726219e-24      # g
GAMMA = 5.0 / 3.0       # Adiabatic index

def gas_temperature(u_code: np.ndarray, xe: np.ndarray) -> np.ndarray:
    """
    Convert IllustrisTNG internal energy (in (km/s)^2) and electron abundance xe to temperature T (K).
    """
    X_H = 0.76
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    u_cgs = u_code * 1e10  # Convert (km/s)^2 to (cm/s)^2
    T = (GAMMA - 1.0) * (u_cgs / KB) * (mu * MP)
    return T

def thermodynamic_entropy_per_particle(T: np.ndarray, rho_code: np.ndarray) -> np.ndarray:
    """
    Calculate specific thermodynamic entropy s ~ log(T) - (gamma - 1) * log(rho).
    """
    s = np.log(np.maximum(T, 1e-10)) - (GAMMA - 1.0) * np.log(np.maximum(rho_code, 1e-30))
    return s

def compute_thermo_entropy(basePath: str, snap: int, subID: int, center: np.ndarray, vel_bulk: np.ndarray, Rmax: float = 200.0, il_module=None) -> dict:
    """
    Load gas snapshot particles and compute mass-weighted thermodynamic entropy statistics.
    """
    if il_module is None:
        import illustris_python as il
    else:
        il = il_module
        
    gas = il.snapshot.loadSubhalo(
        basePath, snap, subID, 'gas',
        fields=['Coordinates', 'Velocities', 'InternalEnergy', 'ElectronAbundance', 'Density', 'Masses']
    )
    
    if gas is None or 'Coordinates' not in gas or len(gas['Coordinates']) == 0:
        return None
        
    pos = gas['Coordinates']
    u = gas['InternalEnergy']
    xe = gas['ElectronAbundance']
    rho = gas['Density']
    mass = gas['Masses']
    
    r = np.linalg.norm(pos - center, axis=1)
    mask = r < Rmax
    
    if np.sum(mask) == 0:
        return None
        
    T = gas_temperature(u[mask], xe[mask])
    s_part = thermodynamic_entropy_per_particle(T, rho[mask])
    w = mass[mask]
    
    mean_S_thermo = float(np.average(s_part, weights=w))
    std_S_thermo = float(np.sqrt(np.average((s_part - mean_S_thermo)**2, weights=w)))
    mean_T = float(np.average(T, weights=w))
    
    return {
        "Mean_Thermo_Entropy": mean_S_thermo,
        "Std_Thermo_Entropy": std_S_thermo,
        "Mean_Temperature": mean_T,
        "Num_Gas_Cells": int(np.sum(mask))
    }
