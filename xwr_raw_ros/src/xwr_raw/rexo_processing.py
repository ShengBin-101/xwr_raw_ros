#!/usr/bin/env python3

"""Range-azimuth and range-elevation heatmaps in the layout of the MMVR dataset, for REXO.

From a reshaped radar cube (chirps x virtual channels x samples):

1. range FFT over the ADC samples and Doppler FFT over the chirps,
2. a Dolph-Chebyshev taper (at least 45 dB) across the antenna elements used,
3. angle FFT over 4 azimuth channels (IWR6843AOP: virtual 11, 9, 7, 5) or 2 elevation
   elements, zero-padded to the output's angle bins, then fftshifted and reversed,
4. the magnitude summed over +-3 Doppler bins around zero velocity,
5. optionally, projection from range-angle (polar) onto a Cartesian grid.

This follows MMVR's description (3D FFT, then integration over Doppler) without being a
copy of its preprocessing. MMVR integrates over the whole Doppler axis. Its maps have 256
range bins, while a profile such as near_mimo.cfg gives 80, which the Cartesian projection
interpolates.

MMVR: M. Mahbubur Rahman et al., "MMVR: Millimeter-wave Multi-View Radar Dataset and Benchmark
for Indoor Perception", ECCV 2024. REXO: https://github.com/merlresearch/radar-bbox-diffusion
"""

import warnings

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from scipy.signal import windows


def fft_angle_bins(n_angle_bins):
    """Angle in radians of each bin of the angle axis that compute_range_*_rexo() produce.

    The angle FFT is fftshifted and then reversed, so bin j sits at
    sin(theta) = 2 (j + 1 - N/2) / N for N bins: uniform in sin(theta), from just above
    -90 degrees to +90 degrees.
    """
    s = 2.0 * (np.arange(n_angle_bins) + 1 - n_angle_bins / 2) / n_angle_bins
    return np.arcsin(np.clip(s, -1.0, 1.0))


def polar_to_cartesian_grid(polar_heatmap,
                            range_max,
                            angle_fov_deg,
                            output_shape=(256, 128),
                            range_bias=0.0,
                            is_horizontal=True,
                            min_range_resolution_m=0.03):
    """
    Project a polar heatmap (range, angle) onto a Cartesian grid.

    The input's angle axis is the one compute_range_*_rexo() produce, uniform in
    sin(theta) over +-90 degrees (fft_angle_bins()). Grid cells outside +-angle_fov_deg/2
    are left at 0.

    Coordinate systems:
        Horizontal view (is_horizontal=True), X-Y seen from above: X lateral, from
        -range_max*sin(FOV/2) to +range_max*sin(FOV/2), Y forward, from 0 to range_max.
        Vertical view (is_horizontal=False): the same grid, with the elevation angle
        measured from the Y axis.
        Row 0 of the output is the far edge (range_max), the last row the radar.

    Args:
        polar_heatmap (np.ndarray): (n_range_bins, n_angle_bins)
        range_max (float): maximum range in metres
        angle_fov_deg (float): full field of view in degrees (120 for +-60)
        output_shape (tuple): (height, width) of the Cartesian grid
        range_bias (float): range of the first range bin in metres
        is_horizontal (bool): True for the horizontal view, False for the vertical view
        min_range_resolution_m (float): the projection runs on a grid at least this fine
            in range and is then resampled to output_shape

    Returns:
        cartesian_heatmap (np.ndarray): float32, output_shape
    """
    n_range_bins, n_angle_bins = polar_heatmap.shape
    out_height, out_width = output_shape

    # Replace unusable geometry with defaults, with a warning, rather than return an
    # empty map.
    if not np.isfinite(range_max) or range_max <= 0:
        warnings.warn(f"polar_to_cartesian_grid: range_max={range_max!r} is unusable, using 10 m")
        range_max = 10.0
    if not np.isfinite(range_bias):
        warnings.warn(f"polar_to_cartesian_grid: range_bias={range_bias!r} is unusable, using 0")
        range_bias = 0.0
    if range_bias >= range_max:
        warnings.warn(f"polar_to_cartesian_grid: range_bias={range_bias!r} >= range_max, using 0")
        range_bias = 0.0
    if not np.isfinite(angle_fov_deg) or angle_fov_deg <= 0:
        warnings.warn(f"polar_to_cartesian_grid: angle_fov_deg={angle_fov_deg!r} is unusable, using 120")
        angle_fov_deg = 120.0
    if not np.isfinite(min_range_resolution_m) or min_range_resolution_m <= 0:
        warnings.warn(f"polar_to_cartesian_grid: min_range_resolution_m={min_range_resolution_m!r} "
                      "is unusable, using 0.03")
        min_range_resolution_m = 0.03

    # Project on a grid fine enough for the native range resolution (about 3 cm on the
    # IWR6843AOP), then resample to output_shape.
    range_span = max(range_max - range_bias, 1e-3)
    internal_height = max(out_height, int(np.ceil(range_span / min_range_resolution_m)))

    # Polar axes
    range_vals = np.linspace(range_bias, range_max, n_range_bins)
    angle_rad_fov = np.deg2rad(angle_fov_deg) / 2  # half the field of view, radians
    angle_vals = fft_angle_bins(n_angle_bins)

    # Physical value -> fractional bin index, -1 outside the axis
    range_to_idx = interp1d(range_vals, np.arange(n_range_bins),
                            bounds_error=False, fill_value=-1)
    angle_to_idx = interp1d(angle_vals, np.arange(n_angle_bins),
                            bounds_error=False, fill_value=-1)

    # Grid with the radar at the bottom middle, far range in row 0. The horizontal and
    # vertical views share it: theta is measured from the Y (forward) axis.
    x_extent = range_max * np.sin(angle_rad_fov)
    x_grid = np.linspace(-x_extent, x_extent, out_width)
    y_grid = np.linspace(range_max, 0, internal_height)
    X, Y = np.meshgrid(x_grid, y_grid)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(X, Y)

    r_indices = range_to_idx(R.ravel())
    theta_indices = angle_to_idx(Theta.ravel())

    valid_mask = (r_indices >= 0) & (r_indices < n_range_bins) & \
                 (theta_indices >= 0) & (theta_indices < n_angle_bins) & \
                 (np.abs(Theta.ravel()) <= angle_rad_fov)

    # No cell inside the polar map (unusable range or field of view): return the polar
    # map's corner unprojected, with a warning, rather than an all-zero frame.
    if not np.any(valid_mask):
        warnings.warn("polar_to_cartesian_grid: no grid cell falls inside the polar map; "
                      "returning it unprojected")
        fallback = np.zeros((internal_height, out_width), dtype=np.float32)
        h = min(internal_height, n_range_bins)
        w = min(out_width, n_angle_bins)
        fallback[:h, :w] = np.nan_to_num(
            polar_heatmap[:h, :w],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)
        if internal_height == out_height:
            return fallback
        src_rows = np.linspace(0.0, 1.0, internal_height)
        dst_rows = np.linspace(0.0, 1.0, out_height)
        resized = np.empty((out_height, out_width), dtype=np.float32)
        for col in range(out_width):
            resized[:, col] = np.interp(dst_rows, src_rows, fallback[:, col])
        return resized

    # Bilinear interpolation of the valid cells
    cartesian_flat = np.zeros(R.size, dtype=polar_heatmap.dtype)
    valid_coords = np.array([r_indices[valid_mask], theta_indices[valid_mask]])
    cartesian_flat[valid_mask] = map_coordinates(
        polar_heatmap,
        valid_coords,
        order=1,
        mode='constant',
        cval=0.0
    )

    cartesian_heatmap = cartesian_flat.reshape(R.shape).astype(np.float32)
    if internal_height == out_height:
        return cartesian_heatmap

    src_rows = np.linspace(0.0, 1.0, internal_height)
    dst_rows = np.linspace(0.0, 1.0, out_height)
    resized = np.empty((out_height, out_width), dtype=np.float32)
    for col in range(out_width):
        resized[:, col] = np.interp(dst_rows, src_rows, cartesian_heatmap[:, col])
    return resized


def condition_heatmap(current_heatmap, previous_heatmap=None, noise_floor=None, alpha=0.5, epsilon=1e-6):
    """
    Radar Heatmap Conditioning compatible with an IWR6843AOP sensor.
    Implements Arc Removal (MVDR Leakage Suppression) and Heatmap Smoothing (Temporal Persistence)
    based on the TI TIDEP-01001 reference design.
    
    Args:
        current_heatmap (np.ndarray): 2D Range-Azimuth Cartesian or Polar heatmap (e.g. 256x128).
        previous_heatmap (np.ndarray, optional): Heatmap from the previous frame for smoothing.
        noise_floor (np.ndarray, optional): Average noise floor row-wise (e.g. shape (256,)).
            If None, defaults to the 10th percentile of each row.
        alpha (float): Persistence factor for slow-fade logic (0.0 to 1.0).
        epsilon (float): Small value for numerical stability to avoid division by zero.
        
    Returns:
        smoothed_heatmap (np.ndarray): Conditioned Heatmap output.
    """
    heatmap = current_heatmap.copy()
    
    # 1. Arc Removal (MVDR Leakage Suppression)
    if noise_floor is None:
        # Default fallback noise estimation if none provided (e.g. 10th percentile per row)
        noise_floor = np.percentile(heatmap, 10, axis=1)
        
    range_row_min_val = np.min(heatmap, axis=1)
    
    # Identify rows where leakage exceeds noise floor
    exceeds_mask = range_row_min_val > noise_floor
    
    if np.any(exceeds_mask):
        row_min = range_row_min_val[exceeds_mask, None]
        row_noise = noise_floor[exceeds_mask, None]
        row_max = np.max(heatmap[exceeds_mask], axis=1, keepdims=True)
        
        # Scale factor to drop leakage to noise floor while maintaining local peaks
        scale_factor = (row_max - row_noise) / (row_max - row_min + epsilon)
        
        # Translate to zero and rescale
        heatmap[exceeds_mask] = (heatmap[exceeds_mask] - row_min) * scale_factor + row_noise
        
    # 2. Heatmap Smoothing (Temporal Persistence)
    if previous_heatmap is not None and previous_heatmap.shape == heatmap.shape:
        # Faded drops: previous multiplied by alpha
        faded_previous = previous_heatmap * alpha
        
        # Sudden spikes appear instantly: we take the maximum between current and faded previous
        smoothed_heatmap = np.maximum(heatmap, faded_previous)
    else:
        smoothed_heatmap = heatmap
        
    return smoothed_heatmap


def compute_range_azimuth_rexo(radar_cube,
                                angle_range=60,
                                n_range_bins=256,
                                n_angle_bins=128,
                                output_cartesian=True,
                                range_max=10.0,
                                range_bias=0.0,
                                range_resolution_m=0.03,
                                window_sidelobe_db=45,
                                platform='xWR68xx_AOP',
                                aop_azimuth_indices=None,
                                doppler_window_bins=3):
    """
    Range-azimuth heatmap for REXO (the horizontal view, MMVR's hm_hori).

    Steps: select the azimuth array, range FFT, Doppler FFT, taper, angle FFT, magnitude
    summed over a window around zero Doppler, and optionally the projection from polar
    (range, azimuth) to a Cartesian grid.

    Args:
        radar_cube (np.ndarray): reshaped cube (n_chirps, n_virtual_rx, n_samples).
            IWR6843AOP with 3 TX: 12 virtual channels (4 RX x 3 TX).
        angle_range (float): half the azimuth field of view in degrees (60 for +-60)
        n_range_bins (int): number of range bins kept (default 256)
        n_angle_bins (int): number of azimuth bins (default 128)
        output_cartesian (bool): if True, project to a Cartesian grid
        range_max (float): maximum range in metres
        range_bias (float): minimum range in metres
        range_resolution_m (float): the profile's range resolution, for the projection
        window_sidelobe_db (float): sidelobe level of the Chebyshev taper, at least 45 dB
        platform (str): radar platform identifier
        aop_azimuth_indices (list[int] | None): the four AOP channels of the azimuth array
        doppler_window_bins (int): half-width of the Doppler window around zero velocity

    Returns:
        range_azimuth (np.ndarray): heatmap, linear magnitude, Cartesian or polar
            (range, azimuth angle)

    Note:
        On the IWR6843AOP (platform containing 'aop', 12 virtual channels) four channels
        form the azimuth array: aop_azimuth_indices, by default [11, 9, 7, 5] (TX3-RX4,
        TX3-RX2, TX2-RX4, TX2-RX2). In that order they sit left to right at 0, lambda/2,
        lambda and 1.5 lambda, a uniform half-wavelength array (antenna layout:
        images/6843aop_ant.png). Within a TX group the channel index does not follow
        position, and the datasheet does not give TX1's horizontal offset, so the other
        channels are left out. Other platforms use every channel in index order.
    """

    n_virtual_rx = radar_cube.shape[1]  # All virtual antennas (e.g., 12 for IWR6843AOP)

    platform_norm = str(platform).lower()
    if 'aop' in platform_norm and n_virtual_rx >= 12:
        if aop_azimuth_indices is None:
            aop_azimuth_indices = [11, 9, 7, 5]
        if len(aop_azimuth_indices) != 4:
            raise ValueError('aop_azimuth_indices must contain exactly 4 virtual channel indices')
        radar_cube = np.stack(
            [radar_cube[:, idx, :] for idx in aop_azimuth_indices],
            axis=1,
        )
        n_virtual_rx = radar_cube.shape[1]
    
    # 1. Range FFT (fast-time dimension / ADC samples)
    range_cube = np.fft.fft(radar_cube, axis=2).astype(np.complex64)
    # Shape: (n_chirps, n_virtual_rx, n_samples)
    
    # 2. Doppler FFT (slow-time dimension / chirps/pulses)
    # This captures velocity information across chirps
    range_doppler = np.fft.fft(range_cube, axis=0).astype(np.complex64)
    # Shape: (n_chirps, n_virtual_rx, n_samples)
    
    # 2.5. Dolph-Chebyshev taper across the array (at least 45 dB sidelobes).
    if not np.isfinite(window_sidelobe_db):
        window_sidelobe_db = 45
    window_sidelobe_db = max(45.0, float(window_sidelobe_db))
    win = windows.chebwin(n_virtual_rx, at=window_sidelobe_db).reshape(1, -1, 1)
    range_doppler_win = range_doppler * win
    
    # 3. Angle FFT over the array, zero-padded to n_angle_bins.
    range_doppler_angle = np.fft.fft(range_doppler_win, n=n_angle_bins, axis=1).astype(np.complex64)
    # Shape: (n_chirps, n_angle_bins, n_samples)
    # fftshift puts boresight in the middle. The bins are then uniform in sin(theta)
    # from -90 to +90 degrees, not in theta (see fft_angle_bins()).
    range_doppler_angle = np.fft.fftshift(range_doppler_angle, axes=1)

    # Reverse the angle axis. np.fft.fft correlates without conjugating, which puts a
    # target at -theta. Reversing the axis is the same as conjugating the input and
    # matches dsp.aoa_capon(), whose matched filter gives +theta. Checked against the
    # camera: a person on the left appears on the left.
    range_doppler_angle = range_doppler_angle[:, ::-1, :]

    # 4. Magnitude summed over a window of +-doppler_window_bins around zero Doppler,
    # not over the whole axis as MMVR does: the targets here (a person, objects on a
    # table) are slow, so the other velocity bins add mostly noise. fftshift puts zero
    # Doppler in the middle.
    range_doppler_angle_shifted = np.fft.fftshift(range_doppler_angle, axes=0)
    n_doppler = range_doppler_angle_shifted.shape[0]
    center = n_doppler // 2
    half_w = max(1, min(int(doppler_window_bins), center))
    dop_lo, dop_hi = center - half_w, center + half_w + 1
    range_angle = np.sum(np.abs(range_doppler_angle_shifted[dop_lo:dop_hi]), axis=0)
    # Shape: (n_angle_bins, n_samples)
    
    # 5. Transpose to (range, angle) format and crop to target range bins
    range_azimuth_polar = range_angle.T[:n_range_bins, :]
    # Shape: (n_range_bins, n_angle_bins) = (256, 128)
    # This is POLAR format: (range, azimuth angle)
    
    # 6. Convert to Cartesian if requested (MMVR standard)
    if output_cartesian:
        range_azimuth_cartesian = polar_to_cartesian_grid(
            range_azimuth_polar,
            range_max=range_max,
            angle_fov_deg=2 * angle_range,  # Full FOV (e.g., 120° for ±60°)
            output_shape=(n_range_bins, n_angle_bins),
            range_bias=range_bias,
            is_horizontal=True,
            min_range_resolution_m=range_resolution_m,
        )
        return range_azimuth_cartesian
    else:
        # Return polar format (for debugging or non-MMVR applications)
        return range_azimuth_polar.astype(np.float32)


def compute_range_elevation_rexo(radar_cube,
                                  platform='xWR68xx',
                                  angle_range=45,
                                  n_range_bins=256,
                                  n_angle_bins=128,
                                  output_cartesian=True,
                                  range_max=10.0,
                                  range_bias=0.0,
                                  range_resolution_m=0.03,
                                  window_sidelobe_db=45,
                                  doppler_window_bins=3):
    """
    Range-elevation heatmap for REXO (the vertical view, MMVR's hm_vert).

    Steps: select the vertical elements, range FFT, Doppler FFT, taper, angle FFT,
    magnitude summed over a window around zero Doppler, and optionally the projection
    from polar (range, elevation) to a Cartesian grid.

    Args:
        radar_cube (np.ndarray): reshaped cube (n_chirps, n_virtual_rx, n_samples).
            IWR6843AOP with 3 TX: 12 virtual channels.
        platform (str): radar platform identifier
        angle_range (float): half the elevation field of view in degrees (45 for +-45)
        n_range_bins (int): number of range bins kept (default 256)
        n_angle_bins (int): number of elevation bins (default 128)
        output_cartesian (bool): if True, project to a Cartesian grid
        range_max (float): maximum range in metres
        range_bias (float): minimum range in metres
        range_resolution_m (float): the profile's range resolution, for the projection
        window_sidelobe_db (float): sidelobe level of the Chebyshev taper, at least 45 dB
        doppler_window_bins (int): half-width of the Doppler window around zero velocity

    Returns:
        range_elevation (np.ndarray): heatmap, linear magnitude, Cartesian or polar
            (range, elevation angle)

    Note:
        Elevation needs channels in one column of the AOP's 2 x 2 receiver grid, at
        different heights: RX1 and RX2 (indices 0 and 1 within a TX group). With 12
        virtual channels, TX2 and TX3 sit at the same height, so their matching channels
        are averaged: [[4, 8], [5, 9]], two elements half a wavelength apart. TX1 is left
        out because the datasheet does not give its horizontal offset. Two elements give a
        broad beam with little height detail, and this view has not been checked against
        ground truth.
    """

    n_virtual_rx = radar_cube.shape[1]

    platform_norm = str(platform).lower()

    # Each group lists channels at the same height. A group's channels are averaged
    # into one element of the vertical array (see the Note).
    if '68xx' in platform_norm and n_virtual_rx >= 12:
        rx_vert_groups = [[4, 8], [5, 9]]  # TX2+TX3 RX0 (Z=0), TX2+TX3 RX1 (Z=lambda/2)
    elif '68xx' in platform_norm and n_virtual_rx >= 8:
        # 8 virtual channels (2 TX): which TXs are active is not known here,
        # so each channel is its own element.
        rx_vert_groups = [[0], [1], [4], [5]]
    elif n_virtual_rx >= 4:
        # No TDM or single TX: use physical RX0 and RX1 (genuine vertical pair)
        rx_vert_groups = [[0], [1]]
    elif n_virtual_rx >= 2:
        # Fallback: use first and last antenna
        rx_vert_groups = [[0], [n_virtual_rx - 1]]
    else:
        # Insufficient antennas for elevation - return zeros
        return np.zeros((n_range_bins, n_angle_bins), dtype=np.float32)

    rx_vert_groups = [[idx for idx in group if idx < n_virtual_rx] for group in rx_vert_groups]
    rx_vert_groups = [group for group in rx_vert_groups if len(group) > 0]
    if len(rx_vert_groups) < 2:
        return np.zeros((n_range_bins, n_angle_bins), dtype=np.float32)

    # Coherently average each group's raw channels: (n_chirps, n_vert, n_samples)
    vertical_data = np.stack(
        [np.mean(radar_cube[:, group, :], axis=1) for group in rx_vert_groups],
        axis=1,
    )
    n_vert = vertical_data.shape[1]

    # If the selected channels have near-zero power, fallback to even-index
    # channels. This helps when virtual channel ordering differs from the
    # assumed TDM mapping.
    vert_power = np.mean(np.abs(vertical_data))
    if not np.isfinite(vert_power) or vert_power < 1e-8:
        alt_indices = list(range(0, n_virtual_rx, 2))[:max(2, min(n_angle_bins, n_virtual_rx // 2))]
        if len(alt_indices) >= 2:
            vertical_data = radar_cube[:, alt_indices, :]
            n_vert = vertical_data.shape[1]
    
    # 1. Range FFT (fast-time dimension / ADC samples)
    range_cube = np.fft.fft(vertical_data, axis=2).astype(np.complex64)
    # Shape: (n_chirps, n_vert, n_samples)
    
    # 2. Doppler FFT (slow-time dimension / chirps/pulses)
    range_doppler = np.fft.fft(range_cube, axis=0).astype(np.complex64)
    # Shape: (n_chirps, n_vert, n_samples)
    
    # 2.5. Dolph-Chebyshev taper across the vertical elements (at least 45 dB sidelobes).
    if not np.isfinite(window_sidelobe_db):
        window_sidelobe_db = 45
    window_sidelobe_db = max(45.0, float(window_sidelobe_db))
    win = windows.chebwin(n_vert, at=window_sidelobe_db).reshape(1, -1, 1) if n_vert > 1 \
        else np.ones((1, 1, 1), dtype=np.float64)
    range_doppler_win = range_doppler * win

    # 3. Angle FFT over the vertical elements, zero-padded to n_angle_bins.
    # Same angle-axis layout as compute_range_azimuth_rexo(): fftshift, then reverse.
    # The reversal there was checked against the camera in azimuth only.
    range_doppler_angle = np.fft.fft(range_doppler_win, n=n_angle_bins, axis=1).astype(np.complex64)
    range_doppler_angle = np.fft.fftshift(range_doppler_angle, axes=1)
    range_doppler_angle = range_doppler_angle[:, ::-1, :]

    # 4. Magnitude summed over the same Doppler window as compute_range_azimuth_rexo().
    range_doppler_angle_shifted = np.fft.fftshift(range_doppler_angle, axes=0)
    n_doppler = range_doppler_angle_shifted.shape[0]
    center = n_doppler // 2
    half_w = max(1, min(int(doppler_window_bins), center))
    dop_lo, dop_hi = center - half_w, center + half_w + 1
    range_angle = np.sum(np.abs(range_doppler_angle_shifted[dop_lo:dop_hi]), axis=0)
    # Shape: (n_angle_bins, n_samples)
    
    # 5. Transpose to (range, angle) format and crop to target range bins
    range_elevation_polar = range_angle.T[:n_range_bins, :]
    # Shape: (n_range_bins, n_angle_bins) = (256, 128)
    # This is POLAR format: (range, elevation angle)
    
    # 6. Convert to Cartesian if requested (MMVR standard)
    if output_cartesian:
        range_elevation_cartesian = polar_to_cartesian_grid(
            range_elevation_polar,
            range_max=range_max,
            angle_fov_deg=2 * angle_range,  # Full FOV (e.g., 90° for ±45°)
            output_shape=(n_range_bins, n_angle_bins),
            range_bias=range_bias,
            is_horizontal=False,  # Vertical view (X-Z plane)
            min_range_resolution_m=range_resolution_m,
        )
        return range_elevation_cartesian
    else:
        # Return polar format (for debugging or non-MMVR applications)
        return range_elevation_polar.astype(np.float32)


def process_radar_cube_for_rexo(radar_cube,
                                 platform='xWR68xx',
                                 azimuth_fov=60,
                                 elevation_fov=45,
                                 n_range_bins=256,
                                 n_azimuth_bins=128,
                                 n_elevation_bins=128,
                                 output_cartesian=True,
                                 range_max=10.0,
                                 range_bias=0.0,
                                 range_resolution_m=0.03,
                                 window_sidelobe_db=45,
                                 doppler_window_bins=3):
    """
    Both REXO views from one radar cube.

    Runs compute_range_azimuth_rexo() and compute_range_elevation_rexo() with shared
    settings and sets NaN and infinite values to 0.

    Args:
        radar_cube (np.ndarray): reshaped cube (n_chirps, n_virtual_rx, n_samples), for
            example from dsp.reshape_frame(). IWR6843AOP with 3 TX: 12 virtual channels.
        platform (str): radar platform, for example 'xWR68xx_AOP'
        azimuth_fov (float): half the horizontal field of view in degrees
        elevation_fov (float): half the vertical field of view in degrees
        n_range_bins, n_azimuth_bins, n_elevation_bins (int): output size
        output_cartesian (bool): True for Cartesian grids, False for polar (range, angle)
        range_max (float): maximum range in metres
        range_bias (float): minimum range in metres
        range_resolution_m (float): the profile's range resolution, for the projection
        window_sidelobe_db (float): sidelobe level of the Chebyshev taper, at least 45 dB
        doppler_window_bins (int): half-width of the Doppler window around zero velocity

    Returns:
        hm_hori, hm_vert (np.ndarray): horizontal and vertical heatmaps, linear magnitude,
            float32. The REXO detector applies its own log scaling and normalisation.
    """
    hm_hori = compute_range_azimuth_rexo(
        radar_cube,
        angle_range=azimuth_fov,
        n_range_bins=n_range_bins,
        n_angle_bins=n_azimuth_bins,
        output_cartesian=output_cartesian,
        range_max=range_max,
        range_bias=range_bias,
        range_resolution_m=range_resolution_m,
        window_sidelobe_db=window_sidelobe_db,
        platform=platform,
        doppler_window_bins=doppler_window_bins,
    )

    hm_vert = compute_range_elevation_rexo(
        radar_cube,
        platform=platform,
        angle_range=elevation_fov,
        n_range_bins=n_range_bins,
        n_angle_bins=n_elevation_bins,
        output_cartesian=output_cartesian,
        range_max=range_max,
        range_bias=range_bias,
        range_resolution_m=range_resolution_m,
        window_sidelobe_db=window_sidelobe_db,
        doppler_window_bins=doppler_window_bins,
    )

    hm_hori = np.nan_to_num(hm_hori, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hm_vert = np.nan_to_num(hm_vert, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return hm_hori, hm_vert


def normalize_for_rexo(heatmap, method='minmax', target_range=(0, 255)):
    """
    Normalize heatmap for REXO model input.
    
    Args:
        heatmap (np.ndarray): Input heatmap (256, 128)
        method (str): Normalization method
            - 'minmax': Scale to [0, 255] or target_range
            - 'zscore': Z-score normalization
            - 'log': Logarithmic scaling (already applied in processing)
        target_range (tuple): Target value range for minmax
    
    Returns:
        normalized (np.ndarray): Normalized heatmap, dtype uint8 or float32
    """
    
    if method == 'minmax':
        # Min-max normalization to target range
        hmin = np.min(heatmap)
        hmax = np.max(heatmap)
        
        if hmax - hmin > 1e-10:
            normalized = (heatmap - hmin) / (hmax - hmin)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = np.zeros_like(heatmap)
        
        return normalized.astype(np.uint8 if target_range == (0, 255) else np.float32)
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(heatmap)
        std = np.std(heatmap)
        
        if std > 1e-10:
            normalized = (heatmap - mean) / std
        else:
            normalized = heatmap - mean
        
        return normalized.astype(np.float32)
    
    else:
        # Return as-is (log scale already applied)
        return heatmap.astype(np.float32)
