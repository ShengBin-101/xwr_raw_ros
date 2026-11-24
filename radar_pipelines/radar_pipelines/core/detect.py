#!/usr/bin/env python3

"""CFAR detections -> 3D points.

The CFAR stage goes through core.cfar.cfar_2d(), so the detector is
selectable. See core/cfar.py's module docstring for why OS-CFAR suits a
tabletop with several objects inside one azimuth beamwidth.
"""

import numpy as np

from .cfar import cfar_2d


def get_cfar_points_3d(range_azimuth, ranges, angles,
                       elevation_map=None,
                       guard_len=2, noise_len=4, threshold=2.5,
                       min_range_idx=0,
                       cfar_method='ca', os_rank_frac=0.75, threshold_pct=99.5,
                       coherence_map=None, min_coherence=0.0,
                       elevation_gate=None):
    """
    Extract 3D points using CFAR from range-azimuth heatmap with elevation.

    Coordinate System (Range-Forward / Y-Forward):
    - X axis: Horizontal (right-left, positive right)
    - Y axis: Forward/Depth (boresight direction)
    - Z axis: Vertical (up-down, positive up)

    Spherical to Cartesian conversion:
    - x = r * sin(azimuth) * cos(elevation)
    - y = r * cos(azimuth) * cos(elevation)
    - z = r * sin(elevation)

    Args:
        range_azimuth (np.ndarray): 2D range-azimuth heatmap (log magnitude)
        ranges (np.ndarray): Range values for each range bin
        angles (np.ndarray): Azimuth angle values (radians) for each azimuth bin
        elevation_map (np.ndarray): Elevation angle in degrees. Either 1D
            (n_range,), one angle per range bin shared by every azimuth at
            that range, which is what xwr_raw/elevation.py's estimators
            produce, or 2D (n_range, n_angle), one angle per cell, as
            core/elevation2d.py produces (see that module). Default None
            means all zeros (flat z=0 output).
        guard_len (int): CFAR guard cells
        noise_len (int): CFAR noise training cells
        threshold (float): CFAR detection threshold (additive, in nats, see core/cfar.py)
        min_range_idx (int): Minimum range bin index to consider
        cfar_method (str): one of core.cfar.CFAR_METHODS: 'ca', 'os', 'go', 'so', 'pct'
        os_rank_frac (float): Order-statistic rank fraction, only used by 'os'
        threshold_pct (float): Percentile threshold, only used by 'pct'
        coherence_map (np.ndarray): Optional (n_range, n_angle) confidence in
            [0,1] accompanying a 2D elevation_map.
        elevation_gate (np.ndarray): Optional (n_range, n_angle) boolean mask
            of cells whose elevation is trusted. ANDed with the coherence
            check. On this hardware the power gate passed here is the one
            that discriminates (see core/elevation2d.py).
        min_coherence (float): Detections whose coherence falls below this get
            elevation forced to 0.0 instead of being dropped. A detection
            with an untrusted height still has a valid range and bearing.
            Points affected are flagged 'elevation_valid': False so a
            consumer can filter them itself. 0.0 disables the check.

    Returns:
        points (list): List of dicts with x, y, z, intensity, range, azimuth, elevation
    """

    # Apply CFAR detection
    mask = cfar_2d(range_azimuth, method=cfar_method, guard_len=guard_len,
                   noise_len=noise_len, threshold=threshold,
                   os_rank_frac=os_rank_frac, threshold_pct=threshold_pct)

    # Filter by min range if needed
    if min_range_idx > 0:
        mask[:min_range_idx, :] = False

    # Elevation may be per range bin (1D) or per cell (2D). Both are
    # accepted, so the xwr_raw estimators work alongside core/elevation2d.py.
    if elevation_map is None:
        elevation_deg = np.zeros(len(ranges), dtype=np.float32)
        elevation_is_2d = False
    else:
        elevation_deg = np.asarray(elevation_map, dtype=np.float32)
        elevation_is_2d = elevation_deg.ndim == 2
        if elevation_is_2d and elevation_deg.shape != range_azimuth.shape:
            raise ValueError(
                f'2D elevation_map shape {elevation_deg.shape} does not match the '
                f'heatmap shape {range_azimuth.shape}. The elevation grid must be '
                f'built on the SAME angle_range/angle_res as the detection heatmap '
                f'or the cells do not correspond.')

    # Get detection indices
    r_indices, a_indices = np.where(mask)
    n_detections = len(r_indices)

    if n_detections == 0:
        return []

    # Vectorized coordinate conversion for efficiency
    r_vals = ranges[r_indices]
    theta_vals = angles[a_indices]  # Azimuth in radians
    if elevation_is_2d:
        elev_vals = elevation_deg[r_indices, a_indices]
    else:
        elev_vals = elevation_deg[r_indices]
    intensities = range_azimuth[r_indices, a_indices]

    # Coherence gate: zero out heights we do not trust, but keep the points.
    elev_valid = np.ones(n_detections, dtype=bool)
    if coherence_map is not None:
        coh = np.asarray(coherence_map, dtype=np.float32)[r_indices, a_indices]
        if min_coherence > 0.0:
            elev_valid &= coh >= float(min_coherence)
    else:
        coh = np.ones(n_detections, dtype=np.float32)
    if elevation_gate is not None:
        elev_valid &= np.asarray(elevation_gate, dtype=bool)[r_indices, a_indices]
    elev_vals = np.where(elev_valid, elev_vals, 0.0)

    phi_vals = np.deg2rad(elev_vals)  # Elevation in radians

    # Pre-compute trig functions
    cos_elev = np.cos(phi_vals)
    sin_elev = np.sin(phi_vals)
    sin_azim = np.sin(theta_vals)
    cos_azim = np.cos(theta_vals)

    # Vectorized spherical to Cartesian
    x_vals = r_vals * sin_azim * cos_elev
    y_vals = r_vals * cos_azim * cos_elev
    z_vals = r_vals * sin_elev

    # Convert to list of dicts
    points = []
    for i in range(n_detections):
        points.append({
            'x': float(x_vals[i]),
            'y': float(y_vals[i]),
            'z': float(z_vals[i]),
            'intensity': float(intensities[i]),
            'range': float(r_vals[i]),
            'azimuth': float(np.rad2deg(theta_vals[i])),
            'elevation': float(elev_vals[i]),
            'elevation_coherence': float(coh[i]),
            'elevation_valid': bool(elev_valid[i]),
        })

    return points
