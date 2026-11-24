#!/usr/bin/env python3

"""Check the interferometric elevation against a target at a known height.

    ros2 run radar_pipelines elevation_calibration.py --true-elevation-deg -12.5

Exploratory. Interferometric elevation (core/elevation2d.py) has never been
checked against ground truth, and it is off by default: the point-cloud
pipeline's elevation_mode defaults to 'none'. This tool is for that check.

Place a strong, isolated target (corner reflector) at a known elevation
angle relative to the radar's boresight, clear everything else you can, and
run this. It takes the strongest cells, reports the elevation the current
settings produce and, if you give it the true angle, prints the
row_phase_offset_deg that would make them agree.

Why it is needed
----------------
core/elevation2d.py's row_phase_offset_deg default (180) comes from a
physical prior, not a measurement: a radar facing down sees the surface
below it at boresight, and 180 is the setting that puts the dominant return
there. That settles the 0-vs-pi question, but it does not confirm the sign
convention (ES1.0 and ES2.0 boards swap RX1/RX2 top-for-bottom, see
xwr_raw/elevation.py's AOP_ELEVATION_SIGN_ES2) and it cannot detect a small
residual bias. Both need a target at a known height.

Run this before trusting absolute elevation numbers. If the reported angle
comes out mirrored about zero, the cause is the board revision, and the fix
is AOP_ELEVATION_SIGN_ES2, not the offset.
"""

import sys
import time
import argparse
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from xwr_raw_ros.msg import RadarFrameFull
from radar_pipelines.core.frame import process_frame_to_range_azimuth
from radar_pipelines.core.elevation2d import estimate_elevation_interferometric


class Collector(Node):
    def __init__(self, n_frames):
        super().__init__('elevation_calibration')
        self.frames = []
        self.n_frames = n_frames
        self.create_subscription(RadarFrameFull, 'radar1/radar_data', self._cb, 5)

    def _cb(self, m):
        if len(self.frames) < self.n_frames:
            self.frames.append(m)

    def done(self):
        return len(self.frames) >= self.n_frames


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--true-elevation-deg', type=float, default=None,
                    help="Target's true elevation relative to radar boresight. "
                         'Omit to just report what the current settings produce.')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--angle-range', type=int, default=70)
    ap.add_argument('--angle-res', type=int, default=1)
    ap.add_argument('--offset-deg', type=float, default=180.0,
                    help='row_phase_offset_deg to evaluate (default: the shipped 180).')
    ap.add_argument('--power-pct', type=float, default=99.9,
                    help='Percentile of heatmap cells treated as "the target".')
    p, _ = ap.parse_known_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=args)
    node = Collector(p.frames)
    node.get_logger().info(
        f'Collecting {p.frames} frames. Keep the target still and the scene otherwise clear.')
    t0 = time.time()
    while rclpy.ok() and not node.done() and time.time() - t0 < 120:
        rclpy.spin_once(node, timeout_sec=0.5)

    if not node.done():
        node.get_logger().error(
            f'Only got {len(node.frames)}/{p.frames} frames. Is the radar publishing?')
        node.destroy_node(); rclpy.shutdown(); return 1

    cubes = deque(maxlen=2)
    per_offset = {p.offset_deg: [], (p.offset_deg + 180.0) % 360.0: []}
    raw_dphi = []
    for m in node.frames:
        _, ra, full = process_frame_to_range_azimuth(m, cubes, p.angle_res, p.angle_range,
                                                     [11, 9, 7, 5], True)
        if ra is None:
            continue
        sel = ra >= np.percentile(ra, p.power_pct)
        for off in per_offset:
            el, _ = estimate_elevation_interferometric(
                full, angle_range=p.angle_range, angle_res=p.angle_res,
                row_phase_offset_deg=off)
            per_offset[off].append(el[sel])
        el0, _ = estimate_elevation_interferometric(
            full, angle_range=p.angle_range, angle_res=p.angle_res, row_phase_offset_deg=0.0)
        raw_dphi.append(np.pi * np.sin(np.deg2rad(el0[sel])))

    print(f'\nStrongest {100 - p.power_pct:.1f}% of cells, {len(node.frames)} frames:\n')
    for off, vals in sorted(per_offset.items()):
        v = np.concatenate(vals)
        tag = '  <- current default' if off == p.offset_deg else ''
        print(f'  row_phase_offset_deg = {off:5.0f} : median {np.median(v):+7.2f} deg   '
              f'p10..p90 {np.percentile(v, 10):+7.2f} .. {np.percentile(v, 90):+7.2f}'
              f'   spread {np.percentile(v, 90) - np.percentile(v, 10):5.1f}{tag}')
    print('\n  A TIGHT spread is the one to believe: a wide spread means dphi has been '
          '\n  pushed near +-pi, where arcsin(dphi/pi) amplifies phase noise without bound.')

    if p.true_elevation_deg is not None:
        v = np.concatenate(per_offset[p.offset_deg])
        measured = float(np.median(v))
        true = p.true_elevation_deg
        # dphi_measured - dphi_true is the residual offset to fold in.
        d_meas = np.median(np.concatenate(raw_dphi))
        d_true = np.pi * np.sin(np.deg2rad(true))
        resid = np.rad2deg(np.angle(np.exp(1j * (d_meas - d_true))))
        print(f'\n  true elevation     : {true:+7.2f} deg')
        print(f'  measured (off={p.offset_deg:.0f}) : {measured:+7.2f} deg   '
              f'error {measured - true:+.2f} deg')
        print(f'\n  Implied row_phase_offset_deg = {resid:+.1f}')
        if abs(measured + true) < abs(measured - true) and abs(true) > 5:
            print('  NOTE: measured is close to the NEGATIVE of true -- that is a '
                  'mirrored\n  sign convention, i.e. an ES1.0 board. Flip '
                  'AOP_ELEVATION_SIGN_ES2 in\n  xwr_raw/elevation.py rather than '
                  'changing the offset.')

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
