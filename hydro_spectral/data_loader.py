"""
Data loader helper wrappers for IllustrisTNG snapshot data and SubLink Merger Trees.
"""

import numpy as np

def load_subhalo_data(basePath: str, snap: int, subhaloID: int, component: str = 'gas', il_module=None) -> dict:
    """
    Load particle positions and bulk-subtracted velocities for a target subhalo.
    """
    if il_module is None:
        import illustris_python as il
    else:
        il = il_module
        
    fields = ['Coordinates', 'Velocities']
    part_data = il.snapshot.loadSubhalo(basePath, snap, subhaloID, component, fields=fields)
    sub = il.groupcat.loadSingle(basePath, snap, subhaloID=subhaloID)
    
    if part_data is None or 'Coordinates' not in part_data or sub is None:
        return None
        
    center = sub['SubhaloPos']
    vel_bulk = sub['SubhaloVel']
    pos = part_data['Coordinates']
    vel = part_data['Velocities'] - vel_bulk
    
    return {
        "pos": pos,
        "vel": vel,
        "center": center,
        "vel_bulk": vel_bulk,
        "subhalo": sub
    }

def load_mpb_branch(basePath: str, snap0: int = 99, target_subhalo: int = 0, snap_stride: int = 5, il_module=None):
    """
    Load the Main Progenitor Branch (MPB) across snapshots using SubLink merger tree.
    """
    if il_module is None:
        import illustris_python as il
    else:
        il = il_module
        
    tree = il.sublink.loadTree(basePath, snap0, target_subhalo, fields=['SnapNum', 'SubfindID', 'MainLeafProgenitorID'])
    if tree is None:
        return [], []
        
    snaps_all = tree['SnapNum']
    subhalos_all = tree['SubfindID']
    
    snaps = snaps_all[::snap_stride]
    subhalos = subhalos_all[::snap_stride]
    
    return snaps, subhalos
