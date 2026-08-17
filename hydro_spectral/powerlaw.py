"""
Power-law spectrum fitting module (E(k) = A * k^alpha) with log-log R^2 optimization.
"""

import numpy as np

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Coefficient of Determination (R^2).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def find_best_linear_region_loglog(
    K: np.ndarray,
    Ek: np.ndarray,
    min_points: int = 4,
    max_points: int = 15
) -> dict:
    """
    Find optimal linear sub-region in log-log space log(E(k)) vs log(k) maximizing R^2.
    
    Parameters
    ----------
    K : np.ndarray
        Wavenumbers k.
    Ek : np.ndarray
        Energy spectrum E(k).
    min_points : int
        Minimum number of points in fitting window.
    max_points : int
        Maximum number of points in fitting window.
        
    Returns
    -------
    dict
        Dictionary containing alpha, logA, R2, k_start, k_end, k_fit, Ek_fit, Ek_fit_line.
    """
    valid = (K > 0) & (Ek > 0) & np.isfinite(K) & np.isfinite(Ek)
    K_val = K[valid]
    Ek_val = Ek[valid]
    
    if len(K_val) < min_points:
        return {
            "alpha": np.nan, "logA": np.nan, "R2": -np.inf,
            "k_start": None, "k_end": None,
            "k_fit": np.array([]), "Ek_fit": np.array([]), "Ek_fit_line": np.array([])
        }
        
    log_k_all = np.log(K_val)
    log_E_all = np.log(Ek_val)
    N = len(K_val)
    
    best_res = None
    best_r2 = -np.inf
    
    max_pts = min(max_points, N)
    for start in range(0, N - min_points + 1):
        for n_pts in range(min_points, max_pts + 1):
            end = start + n_pts
            if end > N:
                break
                
            x = log_k_all[start:end]
            y = log_E_all[start:end]
            
            p = np.polyfit(x, y, 1)
            alpha_candidate = p[0]
            logA_candidate = p[1]
            y_pred = alpha_candidate * x + logA_candidate
            r2 = _r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_res = {
                    "alpha": float(alpha_candidate),
                    "logA": float(logA_candidate),
                    "R2": float(r2),
                    "k_start": float(K_val[start]),
                    "k_end": float(K_val[end - 1]),
                    "k_fit": K_val[start:end],
                    "Ek_fit": Ek_val[start:end],
                    "Ek_fit_line": np.exp(y_pred)
                }
                
    if best_res is None:
        return {
            "alpha": np.nan, "logA": np.nan, "R2": -np.inf,
            "k_start": None, "k_end": None,
            "k_fit": np.array([]), "Ek_fit": np.array([]), "Ek_fit_line": np.array([])
        }
        
    return best_res
