# -*- coding: utf-8 -*-
"""Description of D-infinity (Dinf) flow direction type and methods to convert to/from general 
nextidx, compatible with PyFlwdir."""

from numba import njit, vectorize
import numpy as np
from . import core

__all__ = []

# Dinf type
_ftype = "dinf"
_mv = np.float32(-1)  # No data value
_pv = np.float32(-2)  # Pit value

def calc_dinf_angles(dem, dx=1, dy=1):
    """Calculate D-infinity flow directions in degrees."""
    rows, cols = dem.shape
    angles = np.full((rows, cols), _mv, dtype=np.float32)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if dem[i, j] == _mv:
                continue
            
            # Calculate slopes to 8 neighbors
            s = np.zeros(8, dtype=np.float32)
            s[0] = (dem[i, j] - dem[i, j+1]) / dx if dx != 0 else 0
            s[1] = (dem[i, j] - dem[i-1, j+1]) / np.sqrt(dx**2 + dy**2) if dx != 0 and dy != 0 else 0
            s[2] = (dem[i, j] - dem[i-1, j]) / dy if dy != 0 else 0
            s[3] = (dem[i, j] - dem[i-1, j-1]) / np.sqrt(dx**2 + dy**2) if dx != 0 and dy != 0 else 0
            s[4] = (dem[i, j] - dem[i, j-1]) / dx if dx != 0 else 0
            s[5] = (dem[i, j] - dem[i+1, j-1]) / np.sqrt(dx**2 + dy**2) if dx != 0 and dy != 0 else 0
            s[6] = (dem[i, j] - dem[i+1, j]) / dy if dy != 0 else 0
            s[7] = (dem[i, j] - dem[i+1, j+1]) / np.sqrt(dx**2 + dy**2) if dx != 0 and dy != 0 else 0
            
            # Find maximum downslope direction
            max_slope = np.max(s)
            if max_slope <= 0:
                angles[i, j] = _pv  # Pit
            else:
                max_idx = np.argmax(s)
                if max_idx % 2 == 0:
                    angle = max_idx * 45.0
                else:
                    angle1 = max_idx * 45.0
                    angle2 = ((max_idx + 1) % 8) * 45.0
                    if angle2 < angle1:
                        angle2 += 360.0
                    angle = (angle1 + angle2) / 2
                
                # Ensure angle is between 0 and 360 degrees
                if angle < 0:
                    angle += 360.0
                
                angles[i, j] = angle
    
    return angles

@njit
def get_edge(mask, structure=None):
    """Get edge cells of masked regions."""
    if structure is None:
        structure = np.ones((3, 3), dtype=np.bool_)
    edge = np.zeros_like(mask)
    nrow, ncol = mask.shape
    for r in range(nrow):
        for c in range(ncol):
            if mask[r, c]:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if structure[dr+1, dc+1]:
                            rn, cn = r+dr, c+dc
                            if 0 <= rn < nrow and 0 <= cn < ncol:
                                if not mask[rn, cn]:
                                    edge[r, c] = True
                                    break
                    if edge[r, c]:
                        break
    return edge

@njit
def angle_to_drdc(angle):
    """Convert Dinf angle to delta row/col"""
    if angle == _pv:
        return np.float32(0), np.float32(0)
    dr = -np.sin(angle)
    dc = np.cos(angle)
    return np.float32(dr), np.float32(dc)

@njit
def from_array(flwdir, _mv=_mv, dtype=np.intp):
    """Convert 2D Dinf data to 1D next downstream indices"""
    nrow, ncol = flwdir.shape
    flwdir_flat = flwdir.ravel()
    idxs_ds = np.full(flwdir.size, -1, dtype=dtype)
    pits_lst = []
    n = 0
    
    for idx0 in range(flwdir.size):
        if flwdir_flat[idx0] == _mv:
            continue
        if flwdir_flat[idx0] == _pv:
            pits_lst.append(idx0)
            idxs_ds[idx0] = idx0
            n += 1
            continue
        
        dr, dc = angle_to_drdc(flwdir_flat[idx0])
        r0, c0 = idx0 // ncol, idx0 % ncol
        r_ds, c_ds = r0 + dr, c0 + dc
        
        # Determine the two cells that the flow splits between
        r1, c1 = int(np.floor(r_ds)), int(np.floor(c_ds))
        r2, c2 = int(np.ceil(r_ds)), int(np.ceil(c_ds))
        
        if 0 <= r1 < nrow and 0 <= c1 < ncol:
            idx_ds1 = r1 * ncol + c1
            if flwdir_flat[idx_ds1] != _mv:
                idxs_ds[idx0] = idx_ds1
        
        if 0 <= r2 < nrow and 0 <= c2 < ncol and (r2 != r1 or c2 != c1):
            idx_ds2 = r2 * ncol + c2
            if flwdir_flat[idx_ds2] != _mv:
                if idxs_ds[idx0] == -1:
                    idxs_ds[idx0] = idx_ds2
                else:
                    # If flow splits, choose the steeper direction
                    slope1 = (flwdir_flat[idx0] - flwdir_flat[idx_ds1]) / (np.sqrt((r1-r0)**2 + (c1-c0)**2) + 1e-8)
                    slope2 = (flwdir_flat[idx0] - flwdir_flat[idx_ds2]) / (np.sqrt((r2-r0)**2 + (c2-c0)**2) + 1e-8)
                    idxs_ds[idx0] = idx_ds1 if slope1 > slope2 else idx_ds2
        
        if idxs_ds[idx0] == -1:
            pits_lst.append(idx0)
            idxs_ds[idx0] = idx0
        
        n += 1
    
    return idxs_ds, np.array(pits_lst, dtype=dtype), n


@njit
def _downstream_idx(idx0, flwdir_flat, shape, mv=core._mv):
    """Returns linear index of the downstream neighbor; idx0 if at pit"""
    nrow, ncol = shape
    if flwdir_flat[idx0] == _pv:
        return idx0
    r0 = idx0 // ncol
    c0 = idx0 % ncol
    dr, dc = angle_to_drdc(flwdir_flat[idx0])
    r_ds, c_ds = int(r0 + np.round(dr)), int(c0 + np.round(dc))
    if r_ds >= 0 and r_ds < nrow and c_ds >= 0 and c_ds < ncol:  # check bounds
        idx_ds = c_ds + r_ds * ncol
    else:
        idx_ds = mv
    return idx_ds

@njit
def to_array(idxs_ds, shape, mv=core._mv):
    """convert downstream linear indices to dense Dinf raster"""
    ncol = shape[1]
    flwdir = np.full(idxs_ds.size, _mv, dtype=np.float32)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == mv:
            continue
        if idx_ds == idx0:
            flwdir[idx0] = _pv  # Pit
        else:
            dr = (idx_ds // ncol) - (idx0 // ncol)
            dc = (idx_ds % ncol) - (idx0 % ncol)
            angle = np.arctan2(-dr, dc)
            if angle < 0:
                angle += 2 * np.pi
            flwdir[idx0] = angle
    return flwdir.reshape(shape)

def isvalid(flwdir):
    """True if 2D Dinf raster is valid"""
    return (
        isinstance(flwdir, np.ndarray)
        and flwdir.dtype == np.float32
        and flwdir.ndim == 2
        and np.all((flwdir >= 0) & (flwdir < 2*np.pi) | (flwdir == _mv) | (flwdir == _pv))
    )

@njit
def ispit(dd, _pv=_pv):
    """True if Dinf pit"""
    return dd == _pv

@njit
def isnodata(dd, _mv=_mv):
    """True if Dinf nodata"""
    return dd == _mv

@njit
def _upstream_idx(idx0, flwdir_flat, shape, dtype=np.intp):
    """Returns a numpy array (int64) with linear indices of upstream neighbors"""
    nrow, ncol = shape
    r = idx0 // ncol
    c = idx0 % ncol
    idxs_lst = list()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            r_us, c_us = r + dr, c + dc
            if r_us >= 0 and r_us < nrow and c_us >= 0 and c_us < ncol:
                idx = r_us * ncol + c_us
                angle = flwdir_flat[idx]
                if angle != _mv and angle != _pv:
                    dr_flow, dc_flow = angle_to_drdc(angle)
                    r_flow, c_flow = int(r_us + np.round(dr_flow)), int(c_us + np.round(dc_flow))
                    if r_flow == r and c_flow == c:
                        idxs_lst.append(idx)
    return np.array(idxs_lst, dtype=dtype)