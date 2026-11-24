"""Shared radar DSP layer: everything both pipelines need, defined once."""

from .calib import ProfileCalibration, load_calibration, available_profiles
from .cfar import (CFAR_METHODS, cfar_2d, ca_cfar_2d, os_cfar_2d, go_cfar_2d,
                   so_cfar_2d, pct_detector_2d)
from .cluster import cluster_points_kdtree
from .clutter import BackgroundBuffer, SUBTRACT, SUBTRACT_CLIP
from .detect import get_cfar_points_3d
from .viz import clip_and_normalize
from .frame import (
    process_frame_to_range_azimuth,
    get_range_azimuth_axes,
    get_cartesian_grid,
)

__all__ = [
    'ProfileCalibration', 'load_calibration', 'available_profiles',
    'CFAR_METHODS', 'cfar_2d', 'ca_cfar_2d', 'os_cfar_2d', 'go_cfar_2d', 'so_cfar_2d',
    'pct_detector_2d',
    'cluster_points_kdtree',
    'BackgroundBuffer', 'SUBTRACT', 'SUBTRACT_CLIP',
    'get_cfar_points_3d',
    'process_frame_to_range_azimuth', 'get_range_azimuth_axes', 'get_cartesian_grid',
    'clip_and_normalize',
]
