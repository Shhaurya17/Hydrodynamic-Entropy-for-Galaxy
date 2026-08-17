"""
3D velocity grid construction module using nearest-grid-point (NGP) binning and spatial aperture filtering.
"""

import numpy as np

def filter_aperture(pos: np.ndarray, vel: np.ndarray, center: np.ndarray, Rmax: float):
    """
    Filter particles within spherical aperture of radius Rmax around center.
    """
    r = np.linalg.norm(pos - center, axis=1)
    mask = r < Rmax
    return pos[mask], vel[mask], r[mask]

def construct_velocity_grid(
    pos: np.ndarray,
    vel: np.ndarray,
    center: np.ndarray,
    Rmax: float,
    Ng: int = 32,
    h: float = 0.6774,
    min_cells: int = 100
):
    """
    Construct a 3D regular velocity grid of shape (Ng, Ng, Ng, 3) from particle positions and velocities.
    
    Parameters
    ----------
    pos : np.ndarray
        Particle 3D positions array of shape (N, 3).
    vel : np.ndarray
        Particle 3D velocities array of shape (N, 3).
    center : np.ndarray
        3D coordinate center of the bounding box.
    Rmax : float
        Radius of bounding box half-width (box size is 2 * Rmax).
    Ng : int
        Grid size along each 3D spatial dimension.
    h : float
        Hubble parameter (unused placeholder for physical conversions).
    min_cells : int
        Minimum required particles inside aperture.
        
    Returns
    -------
    np.ndarray or None
        vgrid array of shape (Ng, Ng, Ng, 3), or None if insufficient particles.
    """
    if len(pos) == 0:
        return None
        
    mask = np.linalg.norm(pos - center, axis=1) < Rmax
    pos_sub = pos[mask]
    vel_sub = vel[mask]
    
    if len(pos_sub) < min_cells:
        return None
        
    grid_count = np.zeros((Ng, Ng, Ng), dtype=np.float64)
    vgrid = np.zeros((Ng, Ng, Ng, 3), dtype=np.float64)
    
    box_min = center - Rmax
    box_size = 2.0 * Rmax
    dx = box_size / Ng
    
    ix = np.clip(np.floor((pos_sub[:, 0] - box_min[0]) / dx).astype(int), 0, Ng - 1)
    iy = np.clip(np.floor((pos_sub[:, 1] - box_min[1]) / dx).astype(int), 0, Ng - 1)
    iz = np.clip(np.floor((pos_sub[:, 2] - box_min[2]) / dx).astype(int), 0, Ng - 1)
    
    np.add.at(grid_count, (ix, iy, iz), 1)
    
    for dim in range(3):
        np.add.at(vgrid[:, :, :, dim], (ix, iy, iz), vel_sub[:, dim])
        
    mask_occupied = grid_count > 0
    for dim in range(3):
        vgrid[:, :, :, dim][mask_occupied] /= grid_count[mask_occupied]
        
    return vgrid

def construct_velocity_grid_cpu(pos, vel, center, Rmax, Ng=32, min_cells=100):
    """
    Alias/wrapper for construct_velocity_grid.
    """
    return construct_velocity_grid(pos, vel, center, Rmax, Ng=Ng, min_cells=min_cells)
