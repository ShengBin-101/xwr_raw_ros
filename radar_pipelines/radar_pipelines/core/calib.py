#!/usr/bin/env python3

"""Per-chirp-profile calibration, loaded from one YAML sidecar per profile.

The range correction (range_true = range_scale * range_reported +
range_offset) and the azimuth correction belong to the chirp profile, not to
a node. Each .cfg in xwr_raw_ros/configs/6843aop/ has a sidecar
config/calib/<profile>.yaml, and every node takes a `chirp_profile`
parameter and loads all four constants from it. No node carries its own
default, so the nodes cannot disagree with each other. The package README
explains where the shipped values come from.

Mismatch guard
--------------
RadarFrameFull carries no profile identifier, so a node cannot tell on its
own that the driver runs a different .cfg than the calibration it loaded.
The frame does carry `range_max`, the raw (uncorrected) value
xwr_raw/radar_config.py computes from the profile's slope and sample rate.
Each sidecar records that value as `raw_range_max_m`, and
check_against_frame() compares the two and logs an error on a mismatch, so
a wrong profile shows up as a log line on the first frame instead of as
wrong ranges. near and near_mimo share a profileCfg line, so they have the
same raw_range_max_m and the guard cannot tell them apart. Their range
calibration is the same, so this does not change any range.
"""

import os
import warnings
from dataclasses import dataclass, field

import yaml

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except ImportError:  # running outside a sourced ROS install
    get_package_share_directory = None


# Relative tolerance for the raw_range_max_m consistency check. 2% is wide
# enough to absorb float32 round-tripping through the RadarFrameFull msg
# and the sidecars' own 4-decimal rounding, and far tighter than the gap
# between the shipped profiles with different profileCfg lines (3.6141 m for
# near and near_mimo, 9.03 m for test).
RANGE_MAX_MATCH_RTOL = 0.02


@dataclass
class ProfileCalibration:
    """The complete set of range and azimuth corrections for one chirp profile."""

    profile: str = 'identity'
    range_scale: float = 1.0
    range_offset: float = 0.0
    azimuth_scale: float = 1.0
    azimuth_offset_deg: float = 0.0
    # Raw (uncorrected) range_max that xwr_raw/radar_config.py computes for
    # this profile, i.e. what arrives in frame.range_max. Used only by
    # check_against_frame(). None disables that check.
    raw_range_max_m: float = None
    # range_scale * raw_range_max_m + range_offset, precomputed for
    # readability and cross-checking. Not used in arithmetic: the scale and
    # offset are the source of truth.
    true_range_max_m: float = None
    range_resolution_m: float = None
    notes: str = ''
    source_path: str = field(default='', repr=False)

    @property
    def is_identity(self):
        return (self.range_scale == 1.0 and self.range_offset == 0.0
                and self.azimuth_scale == 1.0 and self.azimuth_offset_deg == 0.0)

    def apply_range(self, r):
        """range_true = range_scale * range_reported + range_offset."""
        return self.range_scale * r + self.range_offset

    def check_against_frame(self, frame, logger=None):
        """Log an error if frame.range_max disagrees with what this profile expects.

        Returns True when consistent (or when the check is not possible),
        False on a mismatch. See "Mismatch guard" in the module docstring.
        """
        if self.raw_range_max_m is None:
            return True
        observed = float(getattr(frame, 'range_max', float('nan')))
        if not (observed > 0):
            return True
        expected = float(self.raw_range_max_m)
        if abs(observed - expected) <= RANGE_MAX_MATCH_RTOL * expected:
            return True
        if logger is not None:
            logger.error(
                f"CHIRP PROFILE MISMATCH: loaded calibration '{self.profile}' was fitted "
                f"for raw range_max={expected:.4f}m, but the driver is reporting "
                f"range_max={observed:.4f}m. The radar is almost certainly running a "
                f"different .cfg than the calibration you loaded -- every range this node "
                f"reports will be wrong by roughly {abs(observed - expected) / expected:.0%}. "
                f"Fix by setting chirp_profile to match the .cfg passed to radar.launch.py, "
                f"or add a sidecar for it in config/calib/. "
                f"Known profiles: {', '.join(available_profiles()) or '(none found)'}.")
        return False


def calib_search_dirs():
    """Where to look for <profile>.yaml, most specific first.

    RADAR_PIPELINES_CALIB_DIR is searched first, so a calibration can be
    overridden without rebuilding the package.
    """
    dirs = []
    env_dir = os.environ.get('RADAR_PIPELINES_CALIB_DIR')
    if env_dir:
        dirs.append(env_dir)
    if get_package_share_directory is not None:
        try:
            dirs.append(os.path.join(
                get_package_share_directory('radar_pipelines'), 'calib'))
        except PackageNotFoundError:
            warnings.warn(
                'radar_pipelines is not installed in a sourced workspace, so its '
                'share/calib directory is not searched. Calibration sidecars are read '
                'from RADAR_PIPELINES_CALIB_DIR and the source tree only.')
    # Source-tree fallback, so the loader also works when a node is run
    # directly with python3 from the source tree.
    dirs.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'config', 'calib'))
    return [d for d in dirs if os.path.isdir(d)]


def available_profiles():
    seen = []
    for d in calib_search_dirs():
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.yaml') and fn[:-5] not in seen:
                seen.append(fn[:-5])
    return seen


def load_calibration(profile, logger=None):
    """Load config/calib/<profile>.yaml.

    An empty or None profile returns the identity calibration, an explicit
    "no correction applied", and logs a warning. An unknown profile name
    raises FileNotFoundError: a typo in chirp_profile must not fall back to
    the identity without notice.
    """
    if not profile or profile == 'identity':
        if logger is not None:
            logger.warn(
                'chirp_profile is unset -- running with IDENTITY range/azimuth '
                'calibration. Reported ranges will be the driver\'s raw, uncorrected '
                'values (about 26% too far on this hardware, judged against a camera). Set '
                f'chirp_profile to one of: {", ".join(available_profiles()) or "(none found)"}.')
        return ProfileCalibration()

    for d in calib_search_dirs():
        path = os.path.join(d, f'{profile}.yaml')
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        cal = ProfileCalibration(
            profile=str(data.get('profile', profile)),
            range_scale=float(data.get('range_scale', 1.0)),
            range_offset=float(data.get('range_offset', 0.0)),
            azimuth_scale=float(data.get('azimuth_scale', 1.0)),
            azimuth_offset_deg=float(data.get('azimuth_offset_deg', 0.0)),
            raw_range_max_m=_opt_float(data.get('raw_range_max_m')),
            true_range_max_m=_opt_float(data.get('true_range_max_m')),
            range_resolution_m=_opt_float(data.get('range_resolution_m')),
            notes=str(data.get('notes', '')),
            source_path=path,
        )
        if logger is not None:
            logger.info(
                f"Loaded calibration '{cal.profile}' from {path}: "
                f"range_true = {cal.range_scale:.5f} * range_reported + {cal.range_offset:.5f}, "
                f"azimuth_true = {cal.azimuth_scale:.5f} * azimuth_reported "
                f"+ {cal.azimuth_offset_deg:.3f} deg")
        return cal

    raise FileNotFoundError(
        f"No calibration sidecar for chirp_profile='{profile}'. Looked in: "
        f"{', '.join(calib_search_dirs()) or '(no calib dirs found -- is the package built and sourced?)'}. "
        f"Known profiles: {', '.join(available_profiles()) or '(none)'}. "
        f"Set chirp_profile to '' to run uncalibrated on purpose.")


def _opt_float(v):
    return None if v is None else float(v)
