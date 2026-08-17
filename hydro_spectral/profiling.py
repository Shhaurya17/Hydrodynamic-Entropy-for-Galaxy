"""
Profiling, FLOPs estimators, and CPU performance benchmark module.
"""

import time
import numpy as np

def estimate_velocity_grid_flops(n_points: int) -> float:
    return n_points * 25.0

def estimate_fft_flops(ng: int, n_components: int = 3) -> float:
    n_vox = ng ** 3
    return n_components * 5.0 * n_vox * np.log2(n_vox)

def estimate_spectrum_binning_flops(ng: int) -> float:
    n_vox = ng ** 3
    return n_vox * 10.0

def estimate_entropy_flops(nbins: int) -> float:
    return nbins * 6.0

def flops_rating(gflops_per_s: float) -> str:
    if gflops_per_s >= 50.0:
        return "EXCELLENT"
    elif gflops_per_s >= 10.0:
        return "GOOD"
    elif gflops_per_s >= 1.0:
        return "MODERATE"
    else:
        return "NEEDS_OPTIMIZATION"

def nsight_style_rating(stage_times: dict) -> str:
    grid_frac = stage_times.get("GridTimeSec", 0) / max(stage_times.get("TotalTimeSec", 1), 1e-9)
    if grid_frac > 0.7:
        return "MEMORY_BOUND_GRID"
    return "BALANCED"

# Backward compatibility aliases
cpu_flops_rating = flops_rating
cpu_nsight_style_rating = nsight_style_rating

def profile_cpu(pos: np.ndarray, vel: np.ndarray, center: np.ndarray, Rmax: float, Ng: int = 32, min_cells: int = 100) -> dict:
    """
    Profile CPU pipeline timing and compute FLOPs ratings.
    """
    from .grid import construct_velocity_grid
    from .spectrum import compute_energy_spectrum, low_k_fraction
    from .entropy import spectral_entropy
    
    t0 = time.perf_counter()
    grid = construct_velocity_grid(pos, vel, center, Rmax, Ng=Ng, min_cells=min_cells)
    t_grid = time.perf_counter() - t0
    
    if grid is None:
        return None
        
    t1 = time.perf_counter()
    Ek = compute_energy_spectrum(grid, Ng=Ng)
    t_fft = time.perf_counter() - t1
    
    t2 = time.perf_counter()
    S = spectral_entropy(Ek)
    lowk = low_k_fraction(Ek)
    t_entropy = time.perf_counter() - t2
    
    tot_time = t_grid + t_fft + t_entropy
    n_pts = len(pos)
    
    f_grid = estimate_velocity_grid_flops(n_pts)
    f_fft = estimate_fft_flops(Ng)
    f_bin = estimate_spectrum_binning_flops(Ng)
    f_ent = estimate_entropy_flops(len(Ek))
    tot_flops = f_grid + f_fft + f_bin + f_ent
    
    gflops = (tot_flops / 1e9) / max(tot_time, 1e-9)
    
    stage_times = {
        "GridTimeSec": t_grid,
        "FFTTimeSec": t_fft,
        "EntropyTimeSec": t_entropy,
        "TotalTimeSec": tot_time
    }
    
    return {
        "Entropy": S,
        "LowKFrac": lowk,
        "GridTimeSec": t_grid,
        "FFTTimeSec": t_fft,
        "EntropyTimeSec": t_entropy,
        "TotalTimeSec": tot_time,
        "TotalFLOPs": tot_flops,
        "GFLOPs": gflops,
        "FLOPsRating": flops_rating(gflops),
        "NsightRating": nsight_style_rating(stage_times),
        "Ek": Ek
    }

def format_markdown_report(payload: dict) -> str:
    """
    Format profiling payload dict as Markdown report.
    """
    return f"""# Performance Profile Report
- **Total Execution Time**: {payload.get('TotalTimeSec', 0):.4f} s
- **Grid Construction**: {payload.get('GridTimeSec', 0):.4f} s
- **FFT Spectrum Calculation**: {payload.get('FFTTimeSec', 0):.4f} s
- **Spectral Entropy Computation**: {payload.get('EntropyTimeSec', 0):.4f} s
- **GFLOP/s**: {payload.get('GFLOPs', 0):.2f}
- **FLOPs Rating**: {payload.get('FLOPsRating', 'N/A')}
- **Roofline Rating**: {payload.get('NsightRating', 'N/A')}
"""
