#!/usr/bin/env python3

"""Raw frame -> range-azimuth heatmap, and the physical axes for it.

Platform reshape, virtual-array selection and Capon (or Bartlett)
beamforming. get_range_azimuth_axes() takes a ProfileCalibration (see
core/calib.py), so callers apply a profile's whole correction at once.
"""

import numpy as np

import xwr_raw.dsp as dsp

from .calib import ProfileCalibration


def _reshape_by_platform(frame):
    platform = frame.platform
    if 'xWR68xx' in platform:
        if 'AOP' in platform:
            return dsp.reshape_frame(frame, flip_aop_phase=True)
        return dsp.reshape_frame(frame, flip_ods_phase=True)

    if 'xWR18xx' in platform:
        return dsp.reshape_frame(frame)

    raise ValueError(f'Unknown radar platform: {platform}')


def _select_virtual_array(platform, radar_cube, aop_azimuth_indices=None):
    if 'xWR68xx' in platform:
        if 'AOP' in platform:
            if aop_azimuth_indices is None:
                aop_azimuth_indices = [11, 9, 7, 5]
            if len(aop_azimuth_indices) != 4:
                raise ValueError('aop_azimuth_indices must contain exactly 4 virtual channel indices')
            # The default [11,9,7,5] assumes a full 12-channel TDM-MIMO cube
            # (3 TX x 4 RX). A single-TX profile does not produce one. near.cfg
            # is such a profile: channelCfg enables all three TX antennas,
            # but only one chirpCfg (TX0) is defined, so reshape_frame
            # returns 4 virtual channels and these indices run off the end of
            # the array.
            #
            # This raises here with an actionable message instead of an
            # IndexError on every frame. It does not remap onto the channels
            # that exist: which columns of a smaller cube form the azimuth
            # ULA depends on the TX antennas used and on the AOP's antenna
            # geometry, and a guess would give a wrong azimuth axis with no
            # error.
            n_virtual = radar_cube.shape[1]
            if max(aop_azimuth_indices) >= n_virtual:
                raise ValueError(
                    f'aop_azimuth_indices {list(aop_azimuth_indices)} require at least '
                    f'{max(aop_azimuth_indices) + 1} virtual channels, but this frame has '
                    f'{n_virtual}. The radar is running a single-TX chirp profile '
                    f'(near.cfg enables 3 TX in channelCfg but fires only TX1), and the '
                    f'point-cloud pipeline needs TDM-MIMO. Use '
                    f'chirp_profile:=near_mimo (same RF tuning as near.cfg, but a genuine '
                    f'3-chirp TDM sequence), or set aop_azimuth_virtual_rx_indices to four '
                    f'valid indices for this cube if you have derived them.')
            return np.stack(
                [
                    radar_cube[:, aop_azimuth_indices[0], :],
                    radar_cube[:, aop_azimuth_indices[1], :],
                    radar_cube[:, aop_azimuth_indices[2], :],
                    radar_cube[:, aop_azimuth_indices[3], :],
                ],
                axis=1,
            )

        radar_cube = np.stack(
            [
                radar_cube[:, 0, :],
                radar_cube[:, 3, :],
                radar_cube[:, 4, :],
                radar_cube[:, 7, :],
            ],
            axis=1,
        )
        return dsp._tdm(radar_cube, 2, 2)

    if 'xWR18xx' in platform:
        if 'AOP' in platform:
            return radar_cube[:, :4, :] + radar_cube[:, 4:8, :] + radar_cube[:, 8:12, :]

        radar_cube = radar_cube[:, :8, :]
        return dsp._tdm(radar_cube, 2, 4)

    raise ValueError(f'Unknown radar platform: {platform}')


def _compute_range_azimuth(frame, radar_cube, capon_angle_resolution=1, capon_angle_range=90,
                            beamformer='capon'):
    """Range-azimuth heatmap, Capon (MVDR) or Bartlett.

    Bartlett is worth trying on static scenes, and not only because it is
    cheaper. Capon's resolution advantage depends on snapshots with
    independent structure to build a well-conditioned covariance from. For
    static tabletop objects at zero Doppler that independence is largely
    absent, so the resolution advantage does not appear while the numerical
    fragility does. On this radar, near-field Capon output swings from frame
    to frame enough to defeat a 30-frame mean subtraction, while the same
    return is stable (coefficient of variation ~0.001) through a plain FFT
    path.

    Capon is the default. Switch with the beamformer parameter, and re-check
    any azimuth fit if you do.
    """
    if beamformer == 'bartlett':
        return dsp.compute_range_azimuth_bartlett(
            radar_cube, capon_angle_resolution, capon_angle_range)
    if frame.adc_output_fmt > 0:
        return dsp.compute_range_azimuth_capon(
            radar_cube,
            capon_angle_resolution,
            capon_angle_range,
        )
    return dsp.compute_range_azimuth_capon_real(
        radar_cube,
        capon_angle_resolution,
        capon_angle_range,
    )


def process_frame_to_range_azimuth(
    frame,
    radar_cubes,
    capon_angle_resolution=1,
    capon_angle_range=90,
    aop_azimuth_indices=None,
    return_full_cube=False,
    beamformer='capon',
):
    """Processes one radar frame into a range-azimuth heatmap.

    return_full_cube=True also returns the full radar cube (all 12 AOP
    virtual channels, before the azimuth selection) as a third value. The
    elevation estimators need it: they use channels that are not in the
    aop_azimuth_indices subset (default [11,9,7,5]) selected for azimuth
    beamforming. Defaults to False, which returns a 2-tuple.
    """
    radar_cube = _reshape_by_platform(frame)

    radar_cubes.append(radar_cube)
    if len(radar_cubes) < radar_cubes.maxlen:
        return (None, None, None) if return_full_cube else (None, None)

    full_radar_cube = np.concatenate(radar_cubes, axis=0)
    radar_cube = _select_virtual_array(
        frame.platform, full_radar_cube, aop_azimuth_indices=aop_azimuth_indices)
    range_azimuth = _compute_range_azimuth(
        frame,
        radar_cube,
        capon_angle_resolution=capon_angle_resolution,
        capon_angle_range=capon_angle_range,
        beamformer=beamformer,
    )

    if return_full_cube:
        return radar_cube, range_azimuth, full_radar_cube
    return radar_cube, range_azimuth


def get_range_azimuth_axes(frame, range_azimuth, angle_range_deg=90, calibration=None,
                            range_scale=None, range_offset=None,
                            azimuth_scale=None, azimuth_offset_deg=None):
    """Builds the physical range and angle axes for a range-azimuth heatmap.

    Pass `calibration` (a core.calib.ProfileCalibration). The four loose
    float arguments are for direct or test use and, when given, override the
    matching field of `calibration`. Node code uses the calibration object,
    loaded per chirp profile from one YAML sidecar, so all nodes apply the
    same constants.

    range_scale and range_offset apply range_true = range_scale *
    range_reported + range_offset on top of the frame.range_max axis.
    RadarConfig.get_params() (xwr_raw/radar_config.py) computes range_max as
    sample_rate * c / (2 * slope). That is larger than the "Maximum
    unambiguous Range" TI's Visualizer writes in each .cfg header, because TI
    derates the maximum range below full Nyquist for an anti-aliasing margin.
    The shipped sidecars set range_scale to the ratio of the two, with
    range_offset 0 (see the package README). This is not a fit to measured
    distances, and it can differ between chirp profiles, so each profile has
    its own sidecar. Defaults to the identity (no correction).

    azimuth_scale and azimuth_offset_deg apply azimuth_true_deg =
    azimuth_scale * azimuth_reported_deg + azimuth_offset_deg.
    gen_steering_vec() assumes an ideal ULA with broadside exactly at
    azimuth 0, and nothing checks that against the board's mounting. One
    rough measurement against the colour camera suggested a large boresight
    offset on this hardware, but no fit was made, so every shipped sidecar
    uses the identity.
    """
    cal = calibration if calibration is not None else ProfileCalibration()
    r_scale = cal.range_scale if range_scale is None else float(range_scale)
    r_offset = cal.range_offset if range_offset is None else float(range_offset)
    a_scale = cal.azimuth_scale if azimuth_scale is None else float(azimuth_scale)
    a_offset = cal.azimuth_offset_deg if azimuth_offset_deg is None else float(azimuth_offset_deg)

    ranges = np.linspace(frame.range_bias, frame.range_max, range_azimuth.shape[0])
    ranges = r_scale * ranges + r_offset
    angle_range_deg = float(abs(angle_range_deg)) if angle_range_deg is not None else 90.0
    # Ascending, to match dsp.gen_steering_vec(), which fills range_azimuth's
    # angle columns with theta_deg = -ang_est_range + kk * ang_est_resolution
    # (kk = 0..N-1). Consumers of this axis (core/detect.py's
    # get_cfar_points_3d) use it without a flip of their own, so a
    # descending axis would mirror left and right.
    angles_deg = np.linspace(-angle_range_deg, angle_range_deg, range_azimuth.shape[1])
    angles_deg = a_scale * angles_deg + a_offset
    angles = np.deg2rad(angles_deg)
    return ranges, angles


def get_cartesian_grid(range_azimuth, range_max):
    x = np.linspace(0.0, range_max, range_azimuth.shape[0])
    y = np.arange(
        -range_max * np.sin(np.deg2rad(90)),
        range_max * np.sin(np.deg2rad(90)),
        range_max / range_azimuth.shape[0],
    )
    return x, y
