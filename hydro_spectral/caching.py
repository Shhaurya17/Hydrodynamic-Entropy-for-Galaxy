"""
Caching utilities for raw spectra data and power-law fitting results.
"""

import os
import pandas as pd

def cache_tag(CACHE_VERSION: str = "v2", SNAP_STRIDE: int = 5, Ng: int = 32, Rmax: float = 200.0, MIN_CELLS: int = 100) -> str:
    """
    Generate unique cache tag based on parameters.
    """
    return f"{CACHE_VERSION}_stride{SNAP_STRIDE}_Ng{Ng}_R{int(Rmax)}_min{MIN_CELLS}"

def spectra_cache_path(component: str, out_dir: str = "cache_spectra", **kwargs) -> str:
    tag = cache_tag(**kwargs)
    return os.path.join(out_dir, f"spectra_{component}_{tag}.csv")

def fit_cache_path(component: str, out_dir: str = "cache_spectra", **kwargs) -> str:
    tag = cache_tag(**kwargs)
    return os.path.join(out_dir, f"fit_{component}_{tag}.csv")

def spectra_list_to_df(spectra_data: list) -> pd.DataFrame:
    """
    Convert list of spectrum dicts into long-format DataFrame.
    """
    rows = []
    for item in spectra_data:
        snap = item["Snapshot"]
        subid = item["SubhaloID"]
        for k_val, e_val in zip(item["k"], item["Ek"]):
            rows.append({
                "Snapshot": snap,
                "SubhaloID": subid,
                "k": k_val,
                "Ek": e_val
            })
    return pd.DataFrame(rows)

def spectra_df_to_list(df: pd.DataFrame) -> list:
    """
    Convert DataFrame back into list of spectrum dicts.
    """
    spectra_data = []
    if df is None or len(df) == 0:
        return spectra_data
        
    for (snap, subid), group in df.groupby(["Snapshot", "SubhaloID"]):
        group_sorted = group.sort_values("k")
        spectra_data.append({
            "Snapshot": int(snap),
            "SubhaloID": int(subid),
            "k": group_sorted["k"].values,
            "Ek": group_sorted["Ek"].values
        })
    return spectra_data
