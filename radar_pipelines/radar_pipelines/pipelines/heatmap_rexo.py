#!/usr/bin/env python3

"""Pipeline 1: MMVR-layout dual-view heatmaps (REXO input).

Raw ADC cube -> (range-azimuth "hm_hori", range-elevation "hm_vert"), the
256x128 dual-view layout of the MMVR dataset, which the REXO model consumes.
The DSP lives in xwr_raw/rexo_processing.py. This module wraps it with range
calibration, near-field exclusion, static background subtraction and
frame-to-frame conditioning, and runs without rclpy so it can be tested.

Config is passed per call, never cached
---------------------------------------
process() takes its RexoHeatmapConfig as an argument instead of storing it
on the instance. The node rebuilds the config from get_parameter() in every
callback, so `ros2 param set` on a running node takes effect on the next
frame. A node that copies its parameters into attributes at start-up keeps
using the old values, even though `ros2 param get` reports the new ones.

Only frame-to-frame state lives on the instance: the previous heatmaps used
by condition_heatmap(), and the captured backgrounds.
"""

from dataclasses import dataclass

import numpy as np

from xwr_raw.dsp import reshape_frame
from xwr_raw.rexo_processing import process_radar_cube_for_rexo, condition_heatmap

from ..core.clutter import BackgroundBuffer, SUBTRACT_CLIP


@dataclass
class RexoHeatmapConfig:
    azimuth_fov: float = 60.0
    elevation_fov: float = 45.0
    n_range_bins: int = 256
    n_azimuth_bins: int = 128
    n_elevation_bins: int = 128
    range_resolution_m: float = 0.04
    window_sidelobe_db: float = 65.0
    # Doppler integration sums only this many bins on either side of zero
    # Doppler (2N+1 bins in total), where MMVR sums the whole Doppler axis.
    # Static and slow targets (a tabletop object, a person walking) keep
    # their energy near zero Doppler, and summing the target-free bins adds
    # noise and lowers contrast.
    doppler_window_bins: int = 3
    flip_aop_phase: bool = True
    output_cartesian: bool = True
    # Near-field exclusion, in true (calibrated) metres. 0.0 = disabled.
    # See _apply_near_field_exclusion().
    min_range_m: float = 0.0
    background_enabled: bool = False
    background_capture: bool = False
    background_n_frames: int = 30
    # Frame-to-frame arc removal and smoothing (xwr_raw.rexo_processing's
    # condition_heatmap). Set False to compare against the raw output.
    conditioning_enabled: bool = True


@dataclass
class RexoHeatmapResult:
    hm_hori: np.ndarray = None
    hm_vert: np.ndarray = None
    # Pre-conditioning, post-background-subtraction copies. This is the
    # stage a background baseline is captured at and subtracted at, so it is
    # also the stage to inspect when tuning suppression.
    hm_hori_raw: np.ndarray = None
    hm_vert_raw: np.ndarray = None
    radar_cube: np.ndarray = None
    actual_range_max: float = 0.0
    capture_status: str = 'idle'


class RexoHeatmapPipeline:
    def __init__(self, calibration=None, logger=None):
        self.calibration = calibration
        self.logger = logger
        self.bg_hori = BackgroundBuffer('hm_hori', mode=SUBTRACT_CLIP)
        self.bg_vert = BackgroundBuffer('hm_vert', mode=SUBTRACT_CLIP)
        self.prev_hm_hori = None
        self.prev_hm_vert = None

    def process(self, frame, cfg):
        radar_cube = reshape_frame(
            frame,
            flip_ods_phase=False,
            flip_aop_phase=cfg.flip_aop_phase,
        )

        # Apply the profile's range calibration, so the heatmap's range span
        # matches what a downstream consumer assumes when it converts
        # heatmap pixels back to metres (see core/calib.py).
        cal = self.calibration
        range_scale = cal.range_scale if cal is not None else 1.0
        range_offset = cal.range_offset if cal is not None else 0.0
        actual_range_max = range_scale * frame.range_max + range_offset
        range_bias_corrected = range_scale * frame.range_bias + range_offset
        range_resolution_m = (
            float(frame.range_res)
            if hasattr(frame, 'range_res') and np.isfinite(frame.range_res) and frame.range_res > 0
            else float(cfg.range_resolution_m)
        )

        hm_hori, hm_vert = process_radar_cube_for_rexo(
            radar_cube,
            platform=frame.platform,
            azimuth_fov=cfg.azimuth_fov,
            elevation_fov=cfg.elevation_fov,
            n_range_bins=cfg.n_range_bins,
            n_azimuth_bins=cfg.n_azimuth_bins,
            n_elevation_bins=cfg.n_elevation_bins,
            output_cartesian=cfg.output_cartesian,
            range_max=actual_range_max,
            range_bias=range_bias_corrected,
            range_resolution_m=range_resolution_m,
            window_sidelobe_db=float(cfg.window_sidelobe_db),
            doppler_window_bins=int(cfg.doppler_window_bins),
        )

        if cfg.min_range_m > 0.0:
            hm_hori = self._apply_near_field_exclusion(
                hm_hori, range_bias_corrected, actual_range_max, cfg.min_range_m,
                cfg.output_cartesian)
            hm_vert = self._apply_near_field_exclusion(
                hm_vert, range_bias_corrected, actual_range_max, cfg.min_range_m,
                cfg.output_cartesian)

        # Background capture and subtraction happen at the same stage, here,
        # before conditioning, so the baseline lines up with the heatmap it
        # is subtracted from (see core/clutter.py).
        status_h = self.bg_hori.handle_capture(
            hm_hori, cfg.background_capture, cfg.background_n_frames, self.logger)
        status_v = self.bg_vert.handle_capture(
            hm_vert, cfg.background_capture, cfg.background_n_frames, self.logger)

        hm_hori = self.bg_hori.apply(hm_hori, cfg.background_enabled, self.logger)
        hm_vert = self.bg_vert.apply(hm_vert, cfg.background_enabled, self.logger)

        result = RexoHeatmapResult(
            hm_hori_raw=hm_hori,
            hm_vert_raw=hm_vert,
            radar_cube=radar_cube,
            actual_range_max=actual_range_max,
            # Both views run the same capture in lockstep. Report 'done' only
            # when both have finished, so the caller resets its trigger once.
            capture_status=status_h if status_h == status_v else 'collecting',
        )

        if cfg.conditioning_enabled:
            # Arc removal & smoothing, using the previous frame as reference.
            hm_hori_c = condition_heatmap(hm_hori, self.prev_hm_hori)
            hm_vert_c = condition_heatmap(hm_vert, self.prev_hm_vert)
        else:
            hm_hori_c, hm_vert_c = hm_hori, hm_vert

        self.prev_hm_hori = hm_hori_c
        self.prev_hm_vert = hm_vert_c

        result.hm_hori = hm_hori_c
        result.hm_vert = hm_vert_c
        return result

    def _apply_near_field_exclusion(self, heatmap, range_bias, range_max, min_range_m,
                                    cartesian):
        # Zeroes rows closer to the sensor than min_range_m. TX/RX near-field
        # self-interference dominates this radar's nearest returns, often
        # above a real target. condition_heatmap()'s arc removal does not
        # remove it: that step only fires when a row's minimum exceeds the
        # noise floor (leakage across the whole row), while this leakage is
        # confined to a few columns near boresight (about 3 of 128).
        #
        # The row-to-range mapping depends on the output mode, and both
        # follow xwr_raw/rexo_processing.py's polar_to_cartesian_grid():
        #   Cartesian  its y_grid: row 0 is range_max (far) and the last row
        #              is 0 (near), whatever range_bias is. A row is a forward
        #              distance Y, and its cells sit at radial range
        #              sqrt(X^2 + Y^2) >= Y, so a near row still holds
        #              returns from farther away at wide angles.
        #   polar      the range axis it assumes for its polar input: row 0
        #              is range_bias (near) and the last row is range_max
        #              (far).
        #
        # min_range_m is a per-profile value. Tune it against live data with
        # the scene cleared, not from this formula alone.
        n_rows = heatmap.shape[0]
        if n_rows <= 1 or not np.isfinite(range_max) or range_max <= 0:
            return heatmap
        if cartesian:
            row_range_m = range_max * (1.0 - np.arange(n_rows) / (n_rows - 1))
        else:
            row_range_m = np.linspace(range_bias, range_max, n_rows)
        mask_rows = row_range_m < min_range_m
        if np.any(mask_rows):
            heatmap = heatmap.copy()
            heatmap[mask_rows, :] = 0.0
        return heatmap
