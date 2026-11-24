#!/usr/bin/env python3

"""Polar and Cartesian heatmap views with real axes, for reading returns by eye.

    ros2 run radar_pipelines heatmap_viz.py --ros-args \
        -p chirp_profile:=near_mimo -p display_range_max:=1.0 \
        -p snapshot_dir:=/tmp/radar_snaps

Publishes four colorized bgr8 images, tick-labelled in metres and degrees:

    radar_viz/range_azimuth              polar, range x azimuth
    radar_viz/range_elevation            polar, range x elevation
    radar_viz/range_azimuth_cartesian    bird's-eye X-Y, radar at bottom centre
    radar_viz/range_elevation_cartesian  side X-Z

How this differs from heatmap_node's _viz topics
------------------------------------------------
heatmap_node publishes the model-facing Cartesian heatmaps, colourized but
unlabelled: enough to confirm data is flowing, not to read a return's
position off the screen. This node is for debugging. It shows polar views
(row N is range bin N, column M is angle bin M, so nothing is interpolated),
Cartesian reprojections of the same arrays, metre and degree ticks on both
axes, and writes throttled PNG snapshots for inspecting a run without a live
viewer.

It drives RexoHeatmapPipeline with output_cartesian=False, so it applies the
chirp profile's range calibration like the rest of the package. It does not
expose near-field exclusion, background subtraction or conditioning, so
those stay off. All settings are ROS parameters, so display_range_max and
the contrast bounds can be changed on a running node.

The polar angle axis keeps every FFT bin. The bins are uniform in
sin(theta) over +-90 deg, not in degrees, so the polar views' angle ticks
are not evenly spaced in degrees (see _fft_angle_deg).
"""

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from xwr_raw_ros.msg import RadarFrameFull
from xwr_raw.rexo_processing import polar_to_cartesian_grid

from radar_pipelines.core.calib import load_calibration
from radar_pipelines.core.viz import clip_and_normalize
from radar_pipelines.pipelines.heatmap_rexo import RexoHeatmapConfig, RexoHeatmapPipeline

# Heatmap cells are 1 px each natively, too small to read without upscaling.
UPSCALE = 4
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _crop_range(hm, range_max, display_range_max):
    """Crop rows to [0, display_range_max], for display only.

    A plain index slice on the calibrated array, not a re-run of the DSP with
    a smaller range_max, which would relabel the bins instead of zooming.
    Range resolution is set by the chirp profile, not by a display parameter.
    """
    if not (display_range_max and 0 < display_range_max < range_max):
        return hm, range_max
    n_keep = max(1, int(round(hm.shape[0] * display_range_max / range_max)))
    return hm[:n_keep], display_range_max


def _canvas(color, title, margin_l=60, margin_t=24, margin_b=20):
    H, W = color.shape[:2]
    c = np.zeros((H + margin_t + margin_b, W + margin_l, 3), dtype=np.uint8)
    c[margin_t:margin_t + H, margin_l:margin_l + W] = color
    cv2.putText(c, title, (margin_l, 16), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return c, H, W, margin_l, margin_t


def _y_ticks(c, H, margin_l, margin_t, value_at):
    for f in _TICKS:
        y = margin_t + int(f * (H - 1))
        cv2.line(c, (margin_l - 5, y), (margin_l, y), (255, 255, 255), 1)
        cv2.putText(c, f'{value_at(f):.2f}m', (2, y + 4), _FONT, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)


def _x_ticks(c, H, W, margin_l, margin_t, value_at, fmt='{:+.0f}', dx=12):
    for f in _TICKS:
        x = margin_l + int(f * (W - 1))
        cv2.line(c, (x, margin_t + H), (x, margin_t + H + 5), (255, 255, 255), 1)
        cv2.putText(c, fmt.format(value_at(f)), (x - dx, margin_t + H + 17), _FONT,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _fft_angle_deg(j, n_bins):
    """Angle in degrees of angle bin j (may be fractional) of a polar REXO heatmap.

    compute_range_*_rexo's angle axis is an FFT, fftshifted and then
    reversed, so its bins are uniform in sin(theta) over +-90 deg, not in
    theta: bin j sits at sin(theta) = 2 * (j + 1 - n_bins / 2) / n_bins.
    """
    s = 2.0 * (j + 1.0 - n_bins / 2.0) / n_bins
    return float(np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0))))


def render_polar(hm, range_max, title, clip, display_range_max=None):
    """(n_range, n_angle) linear magnitude -> tick-labelled BGR image.

    Each x tick is labelled with the true angle of the column under it (see
    _fft_angle_deg). The axis covers all angle bins, so it spans +-90 deg.
    """
    hm, range_max = _crop_range(hm, range_max, display_range_max)
    img8 = clip_and_normalize(np.log1p(hm), *clip)
    color = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    color = cv2.resize(color, (color.shape[1] * UPSCALE, color.shape[0] * UPSCALE),
                       interpolation=cv2.INTER_NEAREST)
    c, H, W, ml, mt = _canvas(color, title)
    # Row 0 = near, last row = range_max.
    _y_ticks(c, H, ml, mt, lambda f: f * range_max)
    n_angle = hm.shape[1]
    _x_ticks(c, H, W, ml, mt, lambda f: _fft_angle_deg(f * (n_angle - 1), n_angle))
    return c


def render_cartesian(polar_hm, range_max, angle_fov, title, clip,
                     output_shape=(500, 400), display_range_max=None):
    """Reproject the same, already computed polar array to a Cartesian view.

    Radar sits bottom-centre (range 0), forward range increases upward,
    lateral position in metres increases rightward. No DSP is re-run.
    """
    polar_hm, range_max = _crop_range(polar_hm, range_max, display_range_max)
    cart = polar_to_cartesian_grid(polar_hm, range_max, 2 * angle_fov,
                                   output_shape=output_shape)
    # Percentile thresholds from the non-zero cells only: the out-of-FoV
    # corners are exact zeros and would otherwise swamp the distribution.
    img8 = clip_and_normalize(np.log1p(cart), *clip, stat_mask=(cart != 0))
    color = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    c, H, W, ml, mt = _canvas(color, title, margin_l=55)
    cv2.drawMarker(c, (ml + W // 2, mt + H - 1), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 10, 2)
    # Image row 0 = range_max (far/top), row H-1 = 0 (near/bottom).
    _y_ticks(c, H, ml, mt, lambda f: (1.0 - f) * range_max)
    x_extent = range_max * np.sin(np.deg2rad(angle_fov))
    _x_ticks(c, H, W, ml, mt, lambda f: -x_extent + f * 2 * x_extent, '{:+.2f}', 16)
    return c


class HeatmapVizNode(Node):
    def __init__(self):
        super().__init__('radar_heatmap_viz')

        self.declare_parameter('chirp_profile', '')
        self.declare_parameter('radar_topic', 'radar1/radar_data')
        self.declare_parameter('azimuth_fov', 60.0)
        self.declare_parameter('elevation_fov', 45.0)
        self.declare_parameter('n_range_bins', 256)
        self.declare_parameter('n_azimuth_bins', 128)
        self.declare_parameter('n_elevation_bins', 128)
        self.declare_parameter('doppler_window_bins', 3)
        # Crop the displayed range (see _crop_range for why this is a slice
        # and not a reprocess). 0.0 = show the full range.
        self.declare_parameter('display_range_max', 0.0)
        self.declare_parameter('viz_clip_low_pct', 2.0)
        self.declare_parameter('viz_clip_high_pct', 98.0)
        self.declare_parameter('snapshot_dir', '')
        self.declare_parameter('snapshot_period_s', 1.0)
        self.declare_parameter('log_period', 25)

        calib = load_calibration(self.get_parameter('chirp_profile').value,
                                 logger=self.get_logger())
        self.pipeline = RexoHeatmapPipeline(calibration=calib, logger=self.get_logger())
        self._calib_checked = False

        self.pubs = {
            'az': self.create_publisher(Image, 'radar_viz/range_azimuth', 1),
            'el': self.create_publisher(Image, 'radar_viz/range_elevation', 1),
            'az_c': self.create_publisher(Image, 'radar_viz/range_azimuth_cartesian', 1),
            'el_c': self.create_publisher(Image, 'radar_viz/range_elevation_cartesian', 1),
        }
        topic = str(self.get_parameter('radar_topic').value)
        self.create_subscription(RadarFrameFull, topic, self.callback, 1)

        self.last_snapshot = 0.0
        self.n_frames = 0
        self.get_logger().info(f'Heatmap viz initialized, subscribed to "{topic}"')

    def callback(self, frame):
        try:
            if not self._calib_checked:
                self.pipeline.calibration.check_against_frame(frame, self.get_logger())
                self._calib_checked = True

            g = self.get_parameter
            az_fov = float(g('azimuth_fov').value)
            el_fov = float(g('elevation_fov').value)
            clip = (float(g('viz_clip_low_pct').value), float(g('viz_clip_high_pct').value))
            drm = float(g('display_range_max').value) or None

            # output_cartesian=False: this node wants the polar arrays, and
            # does its own Cartesian reprojection from them below.
            cfg = RexoHeatmapConfig(
                azimuth_fov=az_fov, elevation_fov=el_fov,
                n_range_bins=int(g('n_range_bins').value),
                n_azimuth_bins=int(g('n_azimuth_bins').value),
                n_elevation_bins=int(g('n_elevation_bins').value),
                doppler_window_bins=int(g('doppler_window_bins').value),
                output_cartesian=False, conditioning_enabled=False)
            res = self.pipeline.process(frame, cfg)
            rmax = res.actual_range_max

            imgs = {
                'az': render_polar(res.hm_hori, rmax, 'Range-Azimuth', clip, drm),
                'el': render_polar(res.hm_vert, rmax, 'Range-Elevation', clip, drm),
                'az_c': render_cartesian(res.hm_hori, rmax, az_fov,
                                         "Range-Azimuth (bird's-eye, X-Y)", clip,
                                         display_range_max=drm),
                'el_c': render_cartesian(res.hm_vert, rmax, el_fov,
                                         'Range-Elevation (side, X-Z)', clip,
                                         display_range_max=drm),
            }
            for k, img in imgs.items():
                self.publish_image(self.pubs[k], img, frame.header)

            self.n_frames += 1
            self._maybe_snapshot(imgs)
            period = max(1, int(g('log_period').value))
            if self.n_frames % period == 0:
                self.get_logger().info(f'frame {self.n_frames}: 4 views published')
        except Exception as e:
            self.get_logger().error(f'frame processing failed: {e}',
                                    throttle_duration_sec=5.0)
            import traceback
            self.get_logger().error(traceback.format_exc(), throttle_duration_sec=5.0)

    def _maybe_snapshot(self, imgs):
        d = str(self.get_parameter('snapshot_dir').value)
        if not d:
            return
        now = time.time()
        if now - self.last_snapshot < float(self.get_parameter('snapshot_period_s').value):
            return
        os.makedirs(d, exist_ok=True)
        names = {'az': 'range_azimuth', 'el': 'range_elevation',
                 'az_c': 'range_azimuth_cartesian', 'el_c': 'range_elevation_cartesian'}
        for k, img in imgs.items():
            cv2.imwrite(os.path.join(d, f'{names[k]}.png'), img)
        self.last_snapshot = now

    def publish_image(self, pub, img_bgr, header):
        msg = Image()
        msg.header = header
        msg.header.frame_id = header.frame_id or 'radar_viz'
        msg.height, msg.width = img_bgr.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img_bgr.tobytes()
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HeatmapVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
