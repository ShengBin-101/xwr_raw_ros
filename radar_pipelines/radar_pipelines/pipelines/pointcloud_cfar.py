#!/usr/bin/env python3

"""Pipeline 2: 3D point cloud via CFAR on the Capon range-azimuth heatmap.

Raw ADC cube -> Capon range-azimuth heatmap -> CFAR detections -> 3D points.
Runs without rclpy, and takes its config per call like the REXO pipeline
(see that module's docstring for why nothing is cached on the instance).

The CFAR stage is selectable: ca, os, go, so or pct (see core/cfar.py). With
several objects inside one azimuth beamwidth of about 25 deg, CA-CFAR's
training cells often include a neighbouring object, which masks the
detection. OS-CFAR is the one to try in that case.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import xwr_raw.elevation as elevation

from ..core.calib import ProfileCalibration
from ..core.cfar import CFAR_METHODS

ELEVATION_MODES = ('none', 'interferometric', 'per_range')
from ..core.clutter import BackgroundBuffer, SUBTRACT
from ..core.detect import get_cfar_points_3d
from ..core.elevation2d import estimate_elevation_interferometric
from ..core.frame import process_frame_to_range_azimuth, get_range_azimuth_axes


@dataclass
class PointCloudConfig:
    # -- beamforming --
    # 'capon' | 'bartlett'. See core/frame.py._compute_range_azimuth.
    beamformer: str = 'capon'
    capon_angle_resolution: int = 1
    capon_angle_range: int = 70
    aop_azimuth_indices: list = field(default_factory=lambda: [11, 9, 7, 5])
    # -- CFAR --
    cfar_method: str = 'ca'
    cfar_threshold: float = 2.5
    cfar_guard_len: int = 2
    cfar_noise_len: int = 4
    cfar_os_rank_frac: float = 0.75
    # 'pct' method only: keep the top (100 - this)% of cells frame-wide.
    cfar_threshold_pct: float = 99.5
    min_range_idx: int = 3
    # -- clutter --
    # Near-field exclusion before CFAR, in true (calibrated) metres. Not the
    # same as min_range_idx, which only filters the final detection mask
    # after CFAR has run over the full array (see
    # _apply_near_field_exclusion()). 0.0 = disabled.
    near_field_exclusion_m: float = 0.0
    background_enabled: bool = True
    background_capture: bool = False
    background_n_frames: int = 30
    # -- elevation --
    # elevation_mode:
    #   'none'            z = 0 for every point (the default).
    #   'interferometric' per-cell elevation from the AOP's two virtual-array
    #                     rows (core/elevation2d.py). Exploratory, not
    #                     checked against ground truth. Needs a TDM-MIMO
    #                     profile (near_mimo): a single TX gives one row and
    #                     no elevation.
    #   'per_range'       xwr_raw/elevation.py: one angle per range bin,
    #                     shared by all azimuths, from a peak search over a
    #                     2-element aperture. The search saturates at the scan
    #                     boundary (in one live frame about half of 63 points
    #                     sat within 0.3 deg of -45 deg). Kept for comparison.
    # Every elevation path here, and the REXO pipeline's hm_vert, has a
    # 2-element vertical aperture, so none can resolve two targets in one
    # cell.
    elevation_mode: str = 'none'
    # Detections whose row-to-row coherence is below this keep their range
    # and bearing but get z = 0 and 'elevation_valid': False. 0.0 disables.
    # Off by default: on live static scenes per-chirp coherence was ~1.00 at
    # every power level (median 0.996 in the bottom 50% of cells), so it does
    # not separate targets from clutter here. It may on a moving or
    # noise-limited scene.
    elevation_min_coherence: float = 0.0
    # Only cells at or above this percentile of the frame's heatmap get a
    # trusted elevation. The rest get z = 0 and 'elevation_valid': False.
    # Cell power tracked phase stability on live data (row-to-row phase
    # concentration 0.988 in the top 0.1% of cells, 0.56-0.79 below).
    # 0 = off.
    elevation_min_power_pct: float = 99.0
    vertical_spacing_lambda: float = 0.5
    # 180 removes the pi that dsp's flip_aop_phase leaves between the two
    # array rows (see core/elevation2d.py).
    elevation_row_phase_offset_deg: float = 180.0
    # 'per_range' mode only.
    elevation_angle_resolution: int = 2
    elevation_angle_range: int = 45
    elevation_method: str = 'bartlett'


@dataclass
class PointCloudResult:
    points: list = field(default_factory=list)
    # The range-azimuth heatmap exactly as CFAR saw it, after clutter
    # suppression and before the CFAR call, in the natural-log magnitude
    # scale core/cfar.py assumes (not dB). Lets threshold, guard_len and
    # noise_len be tuned by eye against the real pre-detection heatmap.
    debug_heatmap: np.ndarray = None
    capture_status: str = 'idle'
    warmup: bool = False


class PointCloudPipeline:
    def __init__(self, calibration=None, logger=None, n_cubes=2):
        self.calibration = calibration if calibration is not None else ProfileCalibration()
        self.logger = logger
        # Two consecutive frames are concatenated before beamforming (see
        # process_frame_to_range_azimuth), so the first callback after
        # start-up has nothing to emit.
        self.radar_cubes = deque(maxlen=n_cubes)
        self.background = BackgroundBuffer('range_azimuth', mode=SUBTRACT)

    def process(self, frame, cfg):
        if cfg.cfar_method not in CFAR_METHODS:
            raise ValueError(
                f"cfar_method '{cfg.cfar_method}' is not one of {CFAR_METHODS}")
        if str(cfg.elevation_mode).lower() not in ELEVATION_MODES:
            raise ValueError(
                f"elevation_mode '{cfg.elevation_mode}' is not one of {ELEVATION_MODES}")

        _, range_azimuth, full_cube = process_frame_to_range_azimuth(
            frame,
            self.radar_cubes,
            capon_angle_resolution=int(cfg.capon_angle_resolution),
            capon_angle_range=int(cfg.capon_angle_range),
            aop_azimuth_indices=[int(x) for x in cfg.aop_azimuth_indices],
            return_full_cube=True,
            beamformer=str(cfg.beamformer),
        )
        if range_azimuth is None:
            return PointCloudResult(warmup=True)

        if cfg.near_field_exclusion_m > 0.0:
            range_azimuth = self._apply_near_field_exclusion(
                range_azimuth, frame, cfg.near_field_exclusion_m)

        status = self.background.handle_capture(
            range_azimuth, cfg.background_capture, cfg.background_n_frames, self.logger)
        range_azimuth = self.background.apply(
            range_azimuth, cfg.background_enabled, self.logger)

        ranges, angles = get_range_azimuth_axes(
            frame,
            range_azimuth,
            angle_range_deg=float(cfg.capon_angle_range),
            calibration=self.calibration,
        )

        elevation_map = None
        coherence_map = None
        mode = str(cfg.elevation_mode).lower()

        if mode == 'interferometric' and full_cube is not None:
            # Built on the same angle grid as the detection heatmap, so cell
            # (r, a) of one indexes cell (r, a) of the other.
            elevation_map, coherence_map = estimate_elevation_interferometric(
                full_cube,
                angle_range=int(cfg.capon_angle_range),
                angle_res=int(cfg.capon_angle_resolution),
                vertical_spacing_lambda=float(cfg.vertical_spacing_lambda),
                row_phase_offset_deg=float(cfg.elevation_row_phase_offset_deg),
            )
            if elevation_map.shape != range_azimuth.shape:
                if self.logger is not None:
                    self.logger.warn(
                        f'elevation grid {elevation_map.shape} != heatmap '
                        f'{range_azimuth.shape}; dropping elevation this frame',
                        throttle_duration_sec=5.0)
                elevation_map = coherence_map = None

        elif mode == 'per_range' and full_cube is not None:
            elevation_map = elevation.estimate_elevation_hybrid(
                full_cube,
                frame.platform,
                angle_res=int(cfg.elevation_angle_resolution),
                angle_range=int(cfg.elevation_angle_range),
                method=str(cfg.elevation_method),
            )
            # estimate_elevation_hybrid returns one angle per range bin
            # (len(ranges)), so every azimuth at a given range shares one
            # elevation. Objects at the same range but different heights
            # cannot both be right.
            if len(elevation_map) != len(ranges):
                if self.logger is not None:
                    self.logger.warn(
                        f'elevation_map length {len(elevation_map)} != ranges length '
                        f'{len(ranges)}, disabling elevation for this frame')
                elevation_map = None

        elevation_gate = None
        if elevation_map is not None and cfg.elevation_min_power_pct > 0:
            thr = np.percentile(range_azimuth, float(cfg.elevation_min_power_pct))
            elevation_gate = range_azimuth >= thr

        points = get_cfar_points_3d(
            range_azimuth,
            ranges,
            angles,
            elevation_map=elevation_map,
            guard_len=int(cfg.cfar_guard_len),
            noise_len=int(cfg.cfar_noise_len),
            threshold=float(cfg.cfar_threshold),
            min_range_idx=int(cfg.min_range_idx),
            cfar_method=str(cfg.cfar_method),
            os_rank_frac=float(cfg.cfar_os_rank_frac),
            threshold_pct=float(cfg.cfar_threshold_pct),
            coherence_map=coherence_map,
            min_coherence=float(cfg.elevation_min_coherence),
            elevation_gate=elevation_gate,
        )

        return PointCloudResult(
            points=points,
            debug_heatmap=range_azimuth,
            capture_status=status,
        )

    def _apply_near_field_exclusion(self, range_azimuth, frame, min_range_m):
        # Replaces range_azimuth rows closer than min_range_m before CFAR
        # runs. A near-field return (about 0.55 m, at boresight) is stable
        # through the REXO pipeline's plain-FFT hm_hori (frame-to-frame
        # coefficient of variation ~0.001) but swings in this Capon path
        # enough to resist a 30-frame mean subtraction. The instability comes
        # from dsp.aoa_capon()'s covariance estimation and inversion, which
        # is sensitive with few chirps and one strong, very close return.
        #
        # min_range_idx cannot do this job. Capon works per range bin, but
        # CFAR's noise estimate for the first reported bins is a spatial
        # average over cells within guard_len + noise_len, which still
        # includes near-field bins. Replacing those rows here, before CFAR,
        # removes them from that average. min_range_idx only filters the
        # mask after CFAR.
        #
        # The row-to-range mapping matches get_range_azimuth_axes(): this
        # array is polar, row 0 is frame.range_bias (near) and the last row
        # is frame.range_max (far), with the calibration applied on top.
        n_rows = range_azimuth.shape[0]
        if n_rows <= 1:
            return range_azimuth
        raw_ranges = np.linspace(frame.range_bias, frame.range_max, n_rows)
        true_ranges = self.calibration.apply_range(raw_ranges)
        mask_rows = true_ranges < min_range_m
        if np.any(mask_rows):
            # Fill with the frame's own mean, not 0.0. This array is
            # natural-log magnitude (unlike REXO's linear hm_hori, where 0.0
            # means "no signal"), and log(1) = 0 has no relation to the
            # noise floor. A zero fill would leave a sharp edge at the
            # exclusion boundary that CFAR's window could read as a
            # detection. The mean blends the excluded rows into the
            # background, as the CFAR kernels assume for non-target cells.
            range_azimuth = range_azimuth.copy()
            fill_value = float(range_azimuth[~mask_rows, :].mean()) if np.any(~mask_rows) \
                else float(range_azimuth.mean())
            range_azimuth[mask_rows, :] = fill_value
        return range_azimuth
