"""
Helmholtz vector field decomposition module (Solenoidal vs Compressive modes).
"""

import numpy as np

def helmholtz_decomposition(vgrid: np.ndarray, Ng: int = None) -> dict:
    """
    Perform 3D Helmholtz vector field decomposition in Fourier space.
    
    Decomposes velocity field v(x) into:
    - Solenoidal (transverse, curl-only): nabla . v_sol = 0
    - Compressive (longitudinal, div-only): nabla x v_comp = 0
    
    Parameters
    ----------
    vgrid : np.ndarray
        3D velocity grid array of shape (Ng, Ng, Ng, 3).
    Ng : int, optional
        Grid resolution along spatial dimension.
        
    Returns
    -------
    dict
        Dictionary containing v_sol, v_comp, P_sol, P_comp, E_sol, E_comp, solenoidal_ratio, compressive_ratio.
    """
    if Ng is None:
        Ng = vgrid.shape[0]
        
    vk_x = np.fft.fftn(vgrid[:, :, :, 0])
    vk_y = np.fft.fftn(vgrid[:, :, :, 1])
    vk_z = np.fft.fftn(vgrid[:, :, :, 2])
    
    kx = np.fft.fftfreq(Ng) * Ng
    ky = np.fft.fftfreq(Ng) * Ng
    kz = np.fft.fftfreq(Ng) * Ng
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero at DC
    
    k_dot_vk = KX * vk_x + KY * vk_y + KZ * vk_z
    
    # Compressive component (parallel to k)
    vk_comp_x = (k_dot_vk / K2) * KX
    vk_comp_y = (k_dot_vk / K2) * KY
    vk_comp_z = (k_dot_vk / K2) * KZ
    
    vk_comp_x[0, 0, 0] = 0.0
    vk_comp_y[0, 0, 0] = 0.0
    vk_comp_z[0, 0, 0] = 0.0
    
    # Solenoidal component (perpendicular to k)
    vk_sol_x = vk_x - vk_comp_x
    vk_sol_y = vk_y - vk_comp_y
    vk_sol_z = vk_z - vk_comp_z
    
    v_sol = np.zeros_like(vgrid)
    v_sol[:, :, :, 0] = np.real(np.fft.ifftn(vk_sol_x))
    v_sol[:, :, :, 1] = np.real(np.fft.ifftn(vk_sol_y))
    v_sol[:, :, :, 2] = np.real(np.fft.ifftn(vk_sol_z))
    
    v_comp = np.zeros_like(vgrid)
    v_comp[:, :, :, 0] = np.real(np.fft.ifftn(vk_comp_x))
    v_comp[:, :, :, 1] = np.real(np.fft.ifftn(vk_comp_y))
    v_comp[:, :, :, 2] = np.real(np.fft.ifftn(vk_comp_z))
    
    P_sol = 0.5 * (np.abs(vk_sol_x)**2 + np.abs(vk_sol_y)**2 + np.abs(vk_sol_z)**2) / (Ng**3)
    P_comp = 0.5 * (np.abs(vk_comp_x)**2 + np.abs(vk_comp_y)**2 + np.abs(vk_comp_z)**2) / (Ng**3)
    
    E_sol = float(np.sum(P_sol))
    E_comp = float(np.sum(P_comp))
    
    E_tot = E_sol + E_comp
    
    return {
        "v_sol": v_sol,
        "v_comp": v_comp,
        "P_sol": P_sol,
        "P_comp": P_comp,
        "E_sol": E_sol,
        "E_comp": E_comp,
        "solenoidal_ratio": E_sol / E_tot if E_tot > 0 else 0.0,
        "compressive_ratio": E_comp / E_tot if E_tot > 0 else 0.0
    }

def power(v: np.ndarray) -> np.ndarray:
    """
    Compute real-space kinetic energy density 0.5 * |v|^2.
    """
    return 0.5 * np.sum(v**2, axis=-1)
