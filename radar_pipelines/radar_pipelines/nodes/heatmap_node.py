#!/usr/bin/env python3

"""ROS 2 node for the REXO/MMVR dual-view heatmap pipeline.

Publishes:
    radar_rexo/hm_hori       32FC1 range-azimuth   (MMVR "horizontal" view)
    radar_rexo/hm_vert       32FC1 range-elevation (MMVR "vertical" view)
    radar_rexo/dual_view     Float32MultiArray, both views stacked
    radar_rexo/hm_*_viz      bgr8 colorized, for eyeballing
    radar_rexo/debug_virtual_channel_power

Parameters are read in every callback into a RexoHeatmapConfig and passed
to the pipeline, so `ros2 param set` on a running node takes effect on the
next frame (see pipelines/heatmap_rexo.py).
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import cv2

from xwr_raw_ros.msg import RadarFrameFull

from radar_pipelines.core.calib import load_calibration
from radar_pipelines.core.viz import clip_and_normalize
from radar_pipelines.pipelines.heatmap_rexo import RexoHeatmapConfig, RexoHeatmapPipeline


class RexoHeatmapNode(Node):
    def __init__(self):
        super().__init__('radar_heatmap_rexo')

        self.bridge = CvBridge()

        # -- calibration ---------------------------------------------------
        # Empty string = run uncalibrated (identity), which load_calibration
        # warns about. An unknown name is an error, not a fallback to the
        # identity (see core/calib.py).
        self.declare_parameter('chirp_profile', '')
        self.declare_parameter('radar_topic', 'radar1/radar_data')

        # -- pipeline ------------------------------------------------------
        self.declare_parameter('azimuth_fov', 60.0)
        self.declare_parameter('elevation_fov', 45.0)
        self.declare_parameter('n_range_bins', 256)
        self.declare_parameter('n_azimuth_bins', 128)
        self.declare_parameter('n_elevation_bins', 128)
        self.declare_parameter('range_resolution_m', 0.04)
        self.declare_parameter('window_sidelobe_db', 65.0)
        self.declare_parameter('doppler_window_bins', 3)
        self.declare_parameter('flip_aop_phase', True)
        self.declare_parameter('output_cartesian', True)
        self.declare_parameter('conditioning_enabled', True)
        # Near-field exclusion in true (calibrated) metres. 0.0 = disabled.
        # The right value depends on the chirp profile, so there is no
        # non-zero default.
        self.declare_parameter('min_range_m', 0.0)

        # -- static background subtraction ---------------------------------
        # Workflow:
        #   1. Remove all objects from the scene (floor/mount/backdrop only).
        #   2. ros2 param set /radar_heatmap_rexo background_capture true
        #   3. Wait for the "Background captured" log (background_n_frames
        #      frames). The trigger resets itself and the baseline is saved to
        #      background_dir when background_autosave is true.
        #   4. Put the objects back.
        # A saved baseline is reloaded at start-up when background_enabled
        # is true, so steps 1-3 survive a node restart.
        self.declare_parameter('background_enabled', False)
        self.declare_parameter('background_capture', False)
        self.declare_parameter('background_n_frames', 30)
        self.declare_parameter('background_dir', self._default_background_dir())
        self.declare_parameter('background_autosave', True)

        # -- output --------------------------------------------------------
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('publish_channel_power_debug', True)
        self.declare_parameter('channel_power_log_period', 50)
        # Display contrast. Plain min/max normalization is dominated by one or
        # two extreme bins (the near-field TX/RX coupling null, or a single
        # very strong clutter return), which compresses everything else into
        # one end of the colour scale. Clipping to these percentiles first
        # spreads the remaining cells over the full scale. These bounds affect
        # the _viz topics only: hm_hori and hm_vert carry the unmodified array.
        self.declare_parameter('viz_clip_low_pct', 2.0)
        self.declare_parameter('viz_clip_high_pct', 98.0)

        calib = load_calibration(
            self.get_parameter('chirp_profile').value, logger=self.get_logger())
        self.pipeline = RexoHeatmapPipeline(calibration=calib, logger=self.get_logger())
        self._calib_checked = False

        bg_dir = self.get_parameter('background_dir').value
        if self.get_parameter('background_enabled').value:
            self.pipeline.bg_hori.load(bg_dir, self.get_logger())
            self.pipeline.bg_vert.load(bg_dir, self.get_logger())

        self.enable_viz = bool(self.get_parameter('enable_visualization').value)
        self.publish_channel_power_debug = bool(
            self.get_parameter('publish_channel_power_debug').value)

        self.hm_hori_pub = self.create_publisher(Image, 'radar_rexo/hm_hori', 1)
        self.hm_vert_pub = self.create_publisher(Image, 'radar_rexo/hm_vert', 1)
        self.dual_view_pub = self.create_publisher(Float32MultiArray, 'radar_rexo/dual_view', 1)

        if self.publish_channel_power_debug:
            self.channel_power_pub = self.create_publisher(
                Float32MultiArray, 'radar_rexo/debug_virtual_channel_power', 1)

        if self.enable_viz:
            self.hm_hori_viz_pub = self.create_publisher(Image, 'radar_rexo/hm_hori_viz', 1)
            self.hm_vert_viz_pub = self.create_publisher(Image, 'radar_rexo/hm_vert_viz', 1)

        topic = str(self.get_parameter('radar_topic').value)
        self.subscription = self.create_subscription(
            RadarFrameFull, topic, self.callback, 1)

        self.frame_count = 0
        self.get_logger().info(
            f'REXO heatmap node initialized, subscribed to "{topic}"')

    @staticmethod
    def _default_background_dir():
        return os.path.join(
            os.path.expanduser('~'), '.ros', 'radar_pipelines', 'background', 'rexo')

    def _read_config(self):
        g = self.get_parameter
        return RexoHeatmapConfig(
            azimuth_fov=float(g('azimuth_fov').value),
            elevation_fov=float(g('elevation_fov').value),
            n_range_bins=int(g('n_range_bins').value),
            n_azimuth_bins=int(g('n_azimuth_bins').value),
            n_elevation_bins=int(g('n_elevation_bins').value),
            range_resolution_m=float(g('range_resolution_m').value),
            window_sidelobe_db=float(g('window_sidelobe_db').value),
            doppler_window_bins=int(g('doppler_window_bins').value),
            flip_aop_phase=bool(g('flip_aop_phase').value),
            output_cartesian=bool(g('output_cartesian').value),
            conditioning_enabled=bool(g('conditioning_enabled').value),
            min_range_m=float(g('min_range_m').value),
            background_enabled=bool(g('background_enabled').value),
            background_capture=bool(g('background_capture').value),
            background_n_frames=int(g('background_n_frames').value),
        )

    def callback(self, frame):
        try:
            if not self._calib_checked:
                # First frame only: the profile-vs-hardware consistency check
                # (see core/calib.py). Once per run is enough, because the
                # driver cannot change chirp profile without restarting.
                self.pipeline.calibration.check_against_frame(frame, self.get_logger())
                self._calib_checked = True

            cfg = self._read_config()
            result = self.pipeline.process(frame, cfg)

            log_period = max(1, int(self.get_parameter('channel_power_log_period').value))
            if self.frame_count % log_period == 0:
                self.get_logger().info(
                    f"[Frame {self.frame_count}] Heatmap Stats (Raw) | "
                    f"Hori Min: {np.min(result.hm_hori):.3e} Max: {np.max(result.hm_hori):.3e} "
                    f"Mean: {np.mean(result.hm_hori):.3e} | "
                    f"Vert Min: {np.min(result.hm_vert):.3e} Max: {np.max(result.hm_vert):.3e} "
                    f"Mean: {np.mean(result.hm_vert):.3e}")
                if self.publish_channel_power_debug and result.radar_cube is not None:
                    self.publish_channel_power(result.radar_cube)

            if result.capture_status == 'done':
                self._finish_background_capture()

            stamp = frame.header.stamp
            self.publish_heatmap(result.hm_hori, self.hm_hori_pub, stamp, 'hori')
            self.publish_heatmap(result.hm_vert, self.hm_vert_pub, stamp, 'vert')
            self.publish_dual_view(result.hm_hori, result.hm_vert, cfg)

            if self.enable_viz:
                self.publish_visualization(result.hm_hori, result.hm_vert, stamp)

            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}', throttle_duration_sec=5.0)
            import traceback
            self.get_logger().error(traceback.format_exc(), throttle_duration_sec=5.0)

    def _finish_background_capture(self):
        # Reset the trigger so the next callback does not start a new
        # capture.
        self.set_parameters([Parameter('background_capture', Parameter.Type.BOOL, False)])
        if bool(self.get_parameter('background_autosave').value):
            bg_dir = str(self.get_parameter('background_dir').value)
            self.pipeline.bg_hori.save(bg_dir, self.get_logger())
            self.pipeline.bg_vert.save(bg_dir, self.get_logger())

    def publish_channel_power(self, radar_cube):
        ch_power = np.mean(np.abs(radar_cube), axis=(0, 2)).astype(np.float32)

        msg = Float32MultiArray()
        dim0 = MultiArrayDimension()
        dim0.label = 'virtual_channel'
        dim0.size = int(ch_power.shape[0])
        dim0.stride = int(ch_power.shape[0])
        msg.layout.dim = [dim0]
        msg.layout.data_offset = 0
        msg.data = ch_power.tolist()

        self.channel_power_pub.publish(msg)

    def publish_heatmap(self, heatmap, publisher, timestamp, view_name):
        img_msg = self.bridge.cv2_to_imgmsg(heatmap.astype(np.float32), encoding='32FC1')
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = f'radar_rexo_{view_name}'
        publisher.publish(img_msg)

    def publish_dual_view(self, hm_hori, hm_vert, cfg):
        dual_view = np.stack([hm_hori, hm_vert], axis=0)

        msg = Float32MultiArray()

        dim0 = MultiArrayDimension()
        dim0.label = 'view'
        dim0.size = 2
        dim0.stride = 2 * cfg.n_range_bins * cfg.n_azimuth_bins

        dim1 = MultiArrayDimension()
        dim1.label = 'range'
        dim1.size = cfg.n_range_bins
        dim1.stride = cfg.n_range_bins * cfg.n_azimuth_bins

        dim2 = MultiArrayDimension()
        dim2.label = 'angle'
        dim2.size = cfg.n_azimuth_bins
        dim2.stride = cfg.n_azimuth_bins

        msg.layout.dim = [dim0, dim1, dim2]
        msg.layout.data_offset = 0
        msg.data = dual_view.flatten().tolist()

        self.dual_view_pub.publish(msg)

    def publish_visualization(self, hm_hori, hm_vert, timestamp):
        lo = float(self.get_parameter('viz_clip_low_pct').value)
        hi = float(self.get_parameter('viz_clip_high_pct').value)
        hm_hori_norm = clip_and_normalize(hm_hori, lo, hi)
        hm_vert_norm = clip_and_normalize(hm_vert, lo, hi)

        hm_hori_color = cv2.applyColorMap(hm_hori_norm, cv2.COLORMAP_JET)
        hm_vert_color = cv2.applyColorMap(hm_vert_norm, cv2.COLORMAP_JET)

        img_msg_hori = self.bridge.cv2_to_imgmsg(hm_hori_color, encoding='bgr8')
        img_msg_hori.header.stamp = timestamp
        img_msg_hori.header.frame_id = 'radar_rexo_hori_viz'
        self.hm_hori_viz_pub.publish(img_msg_hori)

        img_msg_vert = self.bridge.cv2_to_imgmsg(hm_vert_color, encoding='bgr8')
        img_msg_vert.header.stamp = timestamp
        img_msg_vert.header.frame_id = 'radar_rexo_vert_viz'
        self.hm_vert_viz_pub.publish(img_msg_vert)


def main(args=None):
    rclpy.init(args=args)

    node = RexoHeatmapNode()

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
