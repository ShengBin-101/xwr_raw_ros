#!/usr/bin/env python3

"""
Elevation angle estimation for TI mmWave radar platforms.

- IWR6843AOP and IWR6843ODS: 2D antenna array, beamforming over a vertical pair
- IWR1843: linear array, phase difference (little elevation information)
- AWR2944: monopulse when enough antennas are present

Exploratory: these estimates have not been checked against ground truth.
"""

import warnings

import numpy as np
from xwr_raw.dsp import gen_steering_vec, aoa_capon

# Sign applied to the AOP's estimated elevation angle. TI's antenna diagram
# (images/6843aop_ant.png) shows two AOP hardware revisions with RX1 and RX2 swapped:
#   ES1.0: RX1 above RX2 (same column)
#   ES2.0: RX1 below RX2 (same column)
# The vertical pair (RX1/RX2, indices 0 and 1) is the same in both revisions, but the
# sign of the angle flips. The default is ES2.0 (+1). Set -1 if a target at a known
# elevation shows the sign is reversed (an ES1.0 board).
AOP_ELEVATION_SIGN_ES2 = 1


def estimate_elevation_6843aop(radar_cube, angle_res=2, angle_range=45, method='capon'):
    """
    Estimate an elevation angle per range bin on the IWR6843AOP (2 x 2 receiver grid).

    Uses the vertical pair RX1/RX2 (indices 0 and 1 within a TX group), lambda/2 apart.
    With 12 virtual channels, TX2's and TX3's pairs are averaged (see the comment below).

    Args:
        radar_cube (np.ndarray): radar cube (n_chirps, n_virtual_rx, n_samples)
        angle_res (float): elevation angle resolution in degrees
        angle_range (float): elevation angle range (+-angle_range)
        method (str): 'capon' or 'bartlett' beamforming

    Returns:
        elevation_map (np.ndarray): elevation angle in degrees for each range bin
    """
    
    if radar_cube.shape[1] < 2:
        # Need at least 2 antennas for elevation
        return np.zeros(radar_cube.shape[2], dtype=np.float32)

    n_samples = radar_cube.shape[2]
    n_rx = radar_cube.shape[1]

    # The AOP's four receivers form a 2 x 2 grid, lambda/2 apart in both axes.
    # RX1 and RX2 (indices 0 and 1) share a column: the vertical pair, in both
    # hardware revisions. With 12 virtual channels, TX2 and TX3 sit at the same
    # height (the datasheet gives only a horizontal offset of lambda between them),
    # so their matching channels are averaged: TX2-RX1 with TX3-RX1 (4, 8) and
    # TX2-RX2 with TX3-RX2 (5, 9). That gives two elements lambda/2 apart with twice
    # the signal of one TX. TX1 is left out because the datasheet does not give its
    # horizontal offset. Each group in rx_vert_groups is averaged into one element.
    if n_rx >= 12:
        rx_vert_groups = [[4, 8], [5, 9]]
    elif n_rx >= 4:
        rx_vert_groups = [[0], [1]]
    else:
        rx_vert_groups = [[0], [n_rx - 1]]
    n_vert = len(rx_vert_groups)

    # Generate steering vector for elevation
    _, steering_vec_elev = gen_steering_vec(angle_range, angle_res, n_vert)

    # Pre-allocate output array for efficiency
    elevation_angles = np.zeros(n_samples, dtype=np.float32)

    # Process each range bin for elevation
    for r_idx in range(n_samples):
        # Extract vertical antenna data, coherently averaging within each
        # same-Z group: shape (n_vert, n_chirps)
        vert_data = np.stack(
            [np.mean(radar_cube[:, group, r_idx], axis=1) for group in rx_vert_groups],
            axis=0,
        ).astype(np.complex64)
        
        try:
            if method == 'capon':
                # aoa_capon expects shape (num_ant, numChirps)
                elev_spectrum, _ = aoa_capon(vert_data, steering_vec_elev)
                # Find peak in magnitude spectrum
                peak_idx = np.argmax(np.abs(elev_spectrum))
            else:  # bartlett
                # Manual Bartlett: P = |a^H * x|^2, summed over chirps
                # steering_vec_elev: (n_theta, n_ant)
                # vert_data: (n_ant, n_chirps)
                beamformed = np.conj(steering_vec_elev) @ vert_data  # (n_theta, n_chirps)
                elev_spectrum = np.sum(np.abs(beamformed) ** 2, axis=1)  # (n_theta,)
                peak_idx = np.argmax(elev_spectrum)
            
            # Convert bin to angle
            elevation_angles[r_idx] = AOP_ELEVATION_SIGN_ES2 * (-angle_range + peak_idx * angle_res)
            
        except (np.linalg.LinAlgError, ValueError):
            # On numerical error, default to 0 elevation
            elevation_angles[r_idx] = 0.0
    
    return elevation_angles


def estimate_elevation_1843(radar_cube, angle_res=5, angle_range=30):
    """
    Estimate elevation for IWR1843 (linear 3-antenna array).
    
    IWR1843 has 3/4 RX antennas in a linear array.
    Elevation capability is limited compared to 2D arrays.
    Uses phase difference between antenna elements, averaged over chirps.
    
    Args:
        radar_cube (np.ndarray): Radar cube of shape (n_chirps, n_rx, n_samples)
        angle_res (float): Angle resolution in degrees (unused, kept for API consistency)
        angle_range (float): Angle range (±angle_range)
    
    Returns:
        elevation_map (np.ndarray): Estimated elevation angles in degrees
    """
    
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    
    # Pre-allocate output array
    elevation_angles = np.zeros(n_samples, dtype=np.float32)
    
    # Get outer antenna indices
    rx_first = 0
    rx_last = n_rx - 1
    
    # Vectorized computation over all range bins
    # Shape: (n_chirps, n_samples)
    rx_first_data = radar_cube[:, rx_first, :]
    rx_last_data = radar_cube[:, rx_last, :]
    
    # Phase difference for each chirp and range bin
    phase_diff = np.angle(rx_last_data) - np.angle(rx_first_data)
    
    # Normalize to [-pi, pi] and average over chirps
    phase_diff_norm = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    mean_phase_diff = np.mean(phase_diff_norm, axis=0)  # (n_samples,)
    
    # Convert phase difference to elevation angle
    # For antenna spacing d = lambda/2: sin(theta) = phase_diff / pi
    # theta = arcsin(phase_diff / pi) for small angles approximated as linear
    elevation_angles = np.rad2deg(np.arcsin(np.clip(mean_phase_diff / np.pi, -1, 1)))
    elevation_angles = np.clip(elevation_angles, -angle_range, angle_range).astype(np.float32)
    
    return elevation_angles


def estimate_elevation_monopulse(radar_cube, angle_range=45):
    """
    Estimate elevation using monopulse technique.
    
    Monopulse uses sum and difference channels to estimate angle.
    Requires both sum (all antennas) and difference (antenna pairs) channels.
    
    Args:
        radar_cube (np.ndarray): Radar cube (n_chirps, n_rx, n_samples)
        angle_range (float): Angle range
    
    Returns:
        elevation_map (np.ndarray): Elevation angles in degrees
    """
    
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    elevation_angles = np.zeros(n_samples, dtype=np.float32)
    
    for r_idx in range(n_samples):
        # Sum all antennas across chirps: shape (n_rx,) after mean
        slice_data = radar_cube[:, :, r_idx]  # (n_chirps, n_rx)
        sum_channel = np.sum(slice_data, axis=1)  # Sum across RX: (n_chirps,)
        
        # Difference channel (e.g., top_antennas - bottom_antennas)
        if n_rx >= 4:
            diff_channel = (slice_data[:, 0] + slice_data[:, 1] - 
                           slice_data[:, 2] - slice_data[:, 3])
        elif n_rx >= 2:
            # For 2-3 antenna case
            diff_channel = slice_data[:, 0] - slice_data[:, -1]
        else:
            elevation_angles[r_idx] = 0.0
            continue
        
        # Average magnitudes across chirps
        sum_magnitude = np.mean(np.abs(sum_channel))
        diff_magnitude = np.mean(np.abs(diff_channel))
        
        if sum_magnitude > 1e-10:
            monopulse_ratio = diff_magnitude / sum_magnitude
            
            # Convert ratio to angle (simplified, needs calibration)
            # arcsin maps [-1, 1] to [-90, 90] degrees
            elevation_rad = np.arcsin(np.clip(monopulse_ratio, -1, 1))
            elevation_angles[r_idx] = np.rad2deg(elevation_rad) * (angle_range / 90.0)
        else:
            elevation_angles[r_idx] = 0.0
    
    return elevation_angles


def estimate_elevation_doppler_based(radar_cube, velocity_max=2.0, angle_range=30):
    """
    Estimate elevation using Doppler-derived phase information.
    
    This method uses the phase progression across chirps to infer elevation.
    Works best when there's target motion.
    
    Note: This is a simplified approach. Full implementation would require
    knowledge of ego-motion and more sophisticated processing.
    
    Args:
        radar_cube (np.ndarray): Radar cube (n_chirps, n_rx, n_samples)
        velocity_max (float): Maximum velocity for reference (unused currently)
        angle_range (float): Output angle range constraint
    
    Returns:
        elevation_map (np.ndarray): Inferred elevation angles in degrees
    """
    
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    elevation_angles = np.zeros(n_samples, dtype=np.float32)
    
    if n_rx < 2:
        return elevation_angles
    
    # Process each range bin
    for r_idx in range(n_samples):
        # Get data across chirps: (n_chirps, n_rx)
        chirp_data = radar_cube[:, :, r_idx]
        
        # Compute phase difference between first and last RX antenna
        # Average across chirps for robustness
        phase_rx0 = np.angle(chirp_data[:, 0])
        phase_rxN = np.angle(chirp_data[:, -1])
        
        # Phase progression
        phase_diff = phase_rxN - phase_rx0
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # Wrap to [-pi, pi]
        
        # Average phase difference
        mean_phase = np.mean(phase_diff)
        
        # Convert to angle (simplified linear approximation)
        elevation_angles[r_idx] = np.rad2deg(np.arcsin(np.clip(mean_phase / np.pi, -1, 1)))
        elevation_angles[r_idx] = np.clip(elevation_angles[r_idx], -angle_range, angle_range)
    
    return elevation_angles


def estimate_elevation_hybrid(radar_cube, platform, angle_res=2, angle_range=45, method='capon'):
    """
    Select and apply elevation estimation based on radar platform.

    Args:
        radar_cube (np.ndarray): Processed radar cube (n_chirps, n_rx, n_samples)
        platform (str): Radar platform name ('xwr6843aop', 'xwr1843', etc.)
        angle_res (float): Angle resolution
        angle_range (float): Angle range
        method (str): 'capon' or 'bartlett', forwarded to estimate_elevation_6843aop
            (only platform this actually varies for -- the AOP's elevation
            array is just 2 elements, see that function's docstring; Capon's
            adaptive matrix inversion is known to get unstable/spurious on
            arrays this small, especially at wide angle_range, which is
            suspected of causing peak-search results to pile up at the
            search boundary rather than settle on a real angle. Bartlett
            (non-adaptive, no matrix inversion) is more numerically robust,
            though it has even coarser resolution -- this is a diagnostic
            comparison, not a fix for the underlying 2-element aperture
            limit).

    Returns:
        elevation_map (np.ndarray): Elevation angles in degrees
    """
    
    # Handle empty or invalid radar cube
    if radar_cube is None or radar_cube.size == 0:
        return np.array([], dtype=np.float32)
    
    if len(radar_cube.shape) != 3:
        return np.zeros(radar_cube.shape[-1] if len(radar_cube.shape) > 0 else 0, dtype=np.float32)
    
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    
    # Need at least 2 antennas for elevation
    if n_rx < 2:
        return np.zeros(n_samples, dtype=np.float32)
    
    # Platform names from RadarConfig look like 'xWR68xx_AOP' or 'xWR18xx'.
    platform_lower = platform.lower() if platform else ''
    is_68xx = '6843' in platform_lower or '68xx' in platform_lower

    try:
        if is_68xx and ('aop' in platform_lower or 'ods' in platform_lower):
            # 2D antenna array platform - use beamforming
            return estimate_elevation_6843aop(radar_cube, angle_res, angle_range, method=method)
        
        elif '1843' in platform_lower or '18xx' in platform_lower:
            # Linear array - use phase difference
            return estimate_elevation_1843(radar_cube, angle_res, angle_range)
        
        elif '2944' in platform_lower or '29xx' in platform_lower:
            # AWR2944 - try monopulse if enough antennas
            if n_rx >= 4:
                return estimate_elevation_monopulse(radar_cube, angle_range)
            else:
                return np.zeros(n_samples, dtype=np.float32)
        
        else:
            # Unknown platform - try phase-based method if possible
            if n_rx >= 2:
                return estimate_elevation_1843(radar_cube, angle_res, angle_range)
            else:
                return np.zeros(n_samples, dtype=np.float32)
                
    except (np.linalg.LinAlgError, ValueError, IndexError) as e:
        warnings.warn(f"estimate_elevation_hybrid: {type(e).__name__}: {e}; returning zeros")
        return np.zeros(n_samples, dtype=np.float32)


# Utility functions for antenna configuration

def get_antenna_spacing(platform):
    """Get typical antenna spacing in wavelengths for given platform."""
    platform = platform.lower()
    
    spacing = {
        'xwr6843aop': {'horizontal': 0.5, 'vertical': 0.5},  # λ/2 spacing
        'xwr1843': {'linear': 0.5},
        'xwr6843ods': {'horizontal': 0.5, 'vertical': 0.5},
        'awr2944': {'horizontal': 0.5, 'vertical': 0.5},
    }
    
    return spacing.get(platform, {'horizontal': 0.5, 'vertical': 0.5})


def get_antenna_positions(platform, center_freq_ghz=77):
    """
    Get 3D antenna positions in millimeters for a given platform.
    
    Args:
        platform (str): Radar platform name
        center_freq_ghz (float): Center frequency in GHz
    
    Returns:
        dict: Antenna positions indexed by 'rx' and 'tx'
    """
    
    platform = platform.lower()
    wavelength_mm = (3e8 / (center_freq_ghz * 1e9)) * 1000  # wavelength in mm

    # RX positions CONFIRMED against the AOP's own antenna geometry diagram
    # (images/6843aop_ant.png, bundled in this repo, ES2.0 revision): the 4
    # physical RX antennas form a 2x2 grid, lambda/2 spacing in both axes.
    # Coordinates use [X=azimuth (right+), Y=depth (0, same PCB plane),
    # Z=elevation (up+)], origin at RX1 (bottom-right in ES2.0).
    # TX positions are NOT independently re-verified pixel-by-pixel against
    # the diagram (harder to read precisely) and are not used by the
    # current elevation pipeline -- treat as approximate.
    configs = {
        'xwr6843aop': {
            'rx': [
                [0, 0, 0],                                  # RX1: origin (bottom-right)
                [0, 0, wavelength_mm / 2],                   # RX2: same column, above RX1
                [-wavelength_mm / 2, 0, 0],                  # RX3: same row, left of RX1
                [-wavelength_mm / 2, 0, wavelength_mm / 2],  # RX4: top-left
            ],
            'tx': [
                [wavelength_mm / 4, 0, 0],       # TX: right (approximate, unverified)
            ]
        },
        'xwr6843ods': {
            'rx': [
                [0, 0, 0],                      # RX1
                [wavelength_mm/4, 0, 0],       # RX2
                [0, wavelength_mm/4, 0],       # RX3
                [wavelength_mm/4, wavelength_mm/4, wavelength_mm/4],  # RX4
            ],
            'tx': [
                [0, 0, 0],                      # TX
            ]
        }
    }
    
    return configs.get(platform, configs['xwr6843aop'])


if __name__ == '__main__':
    # Example usage
    print("Elevation Estimation Module")
    print("=" * 70)
    print("\nSupported platforms:")
    print("  - IWR6843AOP: 2D array with 4 RX antennas")
    print("  - IWR1843: Linear 3-antenna array")
    print("  - IWR6843ODS: 2D array")
    print("  - AWR2944: Multiband with elevation")
    print("\nUsage:")
    print("  elevation_map = estimate_elevation_hybrid(radar_cube, 'xwr6843aop')")
    print("\nFor specific implementations:")
    print("  estimate_elevation_6843aop()")
    print("  estimate_elevation_monopulse()")
    print("  estimate_elevation_doppler_based()")
