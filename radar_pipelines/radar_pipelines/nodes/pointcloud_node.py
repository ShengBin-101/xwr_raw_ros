#!/usr/bin/env python3

"""ROS 2 node for the CFAR point-cloud pipeline.

Publishes:
    radar_pointcloud                        sensor_msgs/PointCloud2
    radar_pointcloud/markers                one sphere+label per cluster
    radar_pointcloud/debug_heatmap_pre_cfar the array CFAR actually sees

As in heatmap_node.py, parameters are read in every callback into a
PointCloudConfig, so `ros2 param set` on a running node takes effect on the
next frame.
"""

import os
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from xwr_raw_ros.msg import RadarFrameFull

from radar_pipelines.core.calib import load_calibration
from radar_pipelines.core.cfar import CFAR_METHODS
from radar_pipelines.core.cluster import cluster_points_kdtree
from radar_pipelines.pipelines.pointcloud_cfar import (
    ELEVATION_MODES, PointCloudConfig, PointCloudPipeline)


class RadarPointCloudNode(Node):
    def __init__(self):
        super().__init__('radar_pointcloud')

        self.declare_parameter('chirp_profile', '')
        self.declare_parameter('radar_topic', 'radar1/radar_data')

        # -- CFAR ----------------------------------------------------------
        # cfar_method: 'ca' | 'os' | 'go' | 'so' | 'pct'. See core/cfar.py
        # for what each does and when to pick it. The default is 'ca'. 'os'
        # is the one to try on a tabletop with several objects: a
        # neighbouring object inside the same ~25 deg azimuth beamwidth
        # contaminates CA-CFAR's training window and masks the detection.
        self.declare_parameter('cfar_method', 'ca')
        self.declare_parameter('cfar_threshold', 2.5)
        self.declare_parameter('cfar_guard_len', 2)
        self.declare_parameter('cfar_noise_len', 4)
        self.declare_parameter('cfar_os_rank_frac', 0.75)
        self.declare_parameter('cfar_threshold_pct', 99.5)
        self.declare_parameter('beamformer', 'capon')
        self.declare_parameter('min_range_idx', 3)

        # -- beamforming ---------------------------------------------------
        self.declare_parameter('capon_angle_resolution', 1)
        self.declare_parameter('capon_angle_range', 70)
        self.declare_parameter('aop_azimuth_virtual_rx_indices', [11, 9, 7, 5])

        # -- clutter -------------------------------------------------------
        self.declare_parameter('near_field_exclusion_m', 0.0)
        self.declare_parameter('background_enabled', True)
        self.declare_parameter('background_capture', False)
        self.declare_parameter('background_n_frames', 30)
        self.declare_parameter('background_dir', self._default_background_dir())
        self.declare_parameter('background_autosave', True)

        # -- elevation -----------------------------------------------------
        # 'none' | 'interferometric' | 'per_range'. See core/elevation2d.py.
        #   none             z = 0 for every point (the default).
        #   interferometric  per-cell elevation from both rows of the AOP's
        #                    virtual array. Exploratory, not checked against
        #                    ground truth. Needs a TDM-MIMO chirp profile
        #                    (near_mimo): a single TX gives one row, and one
        #                    row carries no elevation.
        #   per_range        one angle per range bin shared by all azimuths,
        #                    from a peak search that saturates at the scan
        #                    boundary. For comparison only.
        self.declare_parameter('elevation_mode', 'none')
        self.declare_parameter('elevation_min_coherence', 0.0)
        self.declare_parameter('elevation_min_power_pct', 99.0)
        self.declare_parameter('vertical_spacing_lambda', 0.5)
        self.declare_parameter('elevation_row_phase_offset_deg', 180.0)
        self.declare_parameter('elevation_angle_resolution', 2)
        self.declare_parameter('elevation_angle_range', 45)
        self.declare_parameter('elevation_method', 'bartlett')
        # Period, in frames, for the elevation-quality summary log. That line
        # reports what fraction of detections passed the elevation gate and
        # the coherence behind them, which the point cloud does not show. Use
        # it to tune elevation_min_coherence.
        self.declare_parameter('elevation_log_period', 25)

        # -- output --------------------------------------------------------
        # publish_pointcloud() remaps the pipeline's native axes (x azimuth,
        # y range, z elevation) to a camera optical convention (x right,
        # y down, z forward). frame_id must name a TF frame with that
        # convention, for example an identity rotation from the colour
        # camera's optical frame. Nothing in this package publishes that
        # transform.
        self.declare_parameter('frame_id', 'radar_horizontal_link')
        self.declare_parameter('publish_debug_heatmap', True)
        self.declare_parameter('publish_cluster_markers', True)
        self.declare_parameter('cluster_tolerance_m', 0.15)
        self.declare_parameter('cluster_min_points', 2)
        self.declare_parameter('cluster_max_points', 500)

        calib = load_calibration(
            self.get_parameter('chirp_profile').value, logger=self.get_logger())
        self.pipeline = PointCloudPipeline(calibration=calib, logger=self.get_logger())
        self._calib_checked = False

        if self.get_parameter('background_enabled').value:
            self.pipeline.background.load(
                str(self.get_parameter('background_dir').value), self.get_logger())

        self.pointcloud_pub = self.create_publisher(PointCloud2, 'radar_pointcloud', 1)
        # The marker publisher is always created (no subscribers, no cost).
        # publish_cluster_markers is read in each callback instead, so it can
        # be changed on a running node.
        self.marker_pub = self.create_publisher(MarkerArray, 'radar_pointcloud/markers', 1)

        self.publish_debug_heatmap = bool(self.get_parameter('publish_debug_heatmap').value)
        if self.publish_debug_heatmap:
            self.bridge = CvBridge()
            self.debug_heatmap_pub = self.create_publisher(
                Image, 'radar_pointcloud/debug_heatmap_pre_cfar', 1)

        topic = str(self.get_parameter('radar_topic').value)
        self.subscription = self.create_subscription(
            RadarFrameFull, topic, self.callback, 1)

        self.get_logger().info(
            f'Radar point cloud node initialized, subscribed to "{topic}"')

    @staticmethod
    def _default_background_dir():
        return os.path.join(
            os.path.expanduser('~'), '.ros', 'radar_pipelines', 'background', 'pointcloud')

    def _read_config(self):
        g = self.get_parameter
        method = str(g('cfar_method').value).lower()
        if method not in CFAR_METHODS:
            self.get_logger().error(
                f"cfar_method '{method}' is not one of {CFAR_METHODS}; falling back to 'ca'",
                throttle_duration_sec=10.0)
            method = 'ca'
        mode = str(g('elevation_mode').value).lower()
        if mode not in ELEVATION_MODES:
            self.get_logger().error(
                f"elevation_mode '{mode}' is not one of {ELEVATION_MODES}; using 'none'",
                throttle_duration_sec=10.0)
            mode = 'none'
        return PointCloudConfig(
            capon_angle_resolution=int(g('capon_angle_resolution').value),
            capon_angle_range=int(g('capon_angle_range').value),
            aop_azimuth_indices=[int(x) for x in g('aop_azimuth_virtual_rx_indices').value],
            cfar_method=method,
            cfar_threshold=float(g('cfar_threshold').value),
            cfar_guard_len=int(g('cfar_guard_len').value),
            cfar_noise_len=int(g('cfar_noise_len').value),
            cfar_os_rank_frac=float(g('cfar_os_rank_frac').value),
            cfar_threshold_pct=float(g('cfar_threshold_pct').value),
            beamformer=str(g('beamformer').value),
            min_range_idx=int(g('min_range_idx').value),
            near_field_exclusion_m=float(g('near_field_exclusion_m').value),
            background_enabled=bool(g('background_enabled').value),
            background_capture=bool(g('background_capture').value),
            background_n_frames=int(g('background_n_frames').value),
            elevation_mode=mode,
            elevation_min_coherence=float(g('elevation_min_coherence').value),
            elevation_min_power_pct=float(g('elevation_min_power_pct').value),
            vertical_spacing_lambda=float(g('vertical_spacing_lambda').value),
            elevation_row_phase_offset_deg=float(g('elevation_row_phase_offset_deg').value),
            elevation_angle_resolution=int(g('elevation_angle_resolution').value),
            elevation_angle_range=int(g('elevation_angle_range').value),
            elevation_method=str(g('elevation_method').value),
        )

    def callback(self, frame):
        try:
            if not self._calib_checked:
                self.pipeline.calibration.check_against_frame(frame, self.get_logger())
                self._calib_checked = True

            cfg = self._read_config()
            result = self.pipeline.process(frame, cfg)
            if result.warmup:
                return

            if result.capture_status == 'done':
                self._finish_background_capture()

            if self.publish_debug_heatmap and result.debug_heatmap is not None:
                self.publish_heatmap_debug(result.debug_heatmap, frame.header.stamp)

            self._log_elevation_quality(cfg, result.points)

            self.publish_pointcloud(result.points, frame.header.stamp)

            if bool(self.get_parameter('publish_cluster_markers').value):
                self.publish_cluster_markers_msg(result.points, frame.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}', throttle_duration_sec=5.0)
            import traceback
            self.get_logger().error(traceback.format_exc(), throttle_duration_sec=5.0)

    def _log_elevation_quality(self, cfg, points):
        """Periodic summary of how much of the elevation output passed the gate.

        Points whose height was rejected look the same as points really at
        z=0, so without this there is no way to tell 'the scene is flat' from
        'every height was rejected'."""
        period = max(1, int(self.get_parameter('elevation_log_period').value))
        self._elev_frame = getattr(self, '_elev_frame', 0) + 1
        if cfg.elevation_mode == 'none' or not points:
            return
        if self._elev_frame % period:
            return
        coh = np.array([p['elevation_coherence'] for p in points])
        valid = np.array([p['elevation_valid'] for p in points])
        el = np.array([p['elevation'] for p in points])[valid]
        msg = (f'[elevation:{cfg.elevation_mode}] {valid.sum()}/{len(points)} points '
               f'passed the gate (power p{cfg.elevation_min_power_pct:.0f}'
               f'{f", coh {cfg.elevation_min_coherence:.2f}" if cfg.elevation_min_coherence > 0 else ""})'
               f' | coh med {np.median(coh):.2f}')
        if el.size:
            msg += f' | elevation {el.min():+.1f} to {el.max():+.1f} deg, med {np.median(el):+.1f}'
        self.get_logger().info(msg)

    def _finish_background_capture(self):
        self.set_parameters([Parameter('background_capture', Parameter.Type.BOOL, False)])
        if bool(self.get_parameter('background_autosave').value):
            self.pipeline.background.save(
                str(self.get_parameter('background_dir').value), self.get_logger())

    def publish_heatmap_debug(self, range_azimuth, stamp):
        img_msg = self.bridge.cv2_to_imgmsg(range_azimuth.astype(np.float32), encoding='32FC1')
        img_msg.header.stamp = stamp
        # Not the point cloud's frame_id: this is the raw range x azimuth
        # array in its own index space, not a posed 3D frame, so it gets a
        # dedicated frame_id that no TF tree contains.
        img_msg.header.frame_id = 'radar_range_azimuth_debug'
        self.debug_heatmap_pub.publish(img_msg)

    def publish_pointcloud(self, points, stamp):
        if not points:
            return

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.get_parameter('frame_id').value

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        # Axis remap. core/detect.py outputs a radar-native convention
        # (x = azimuth/right, y = range/forward, z = elevation/up). frame_id
        # is expected to use a camera optical convention (x = right,
        # y = down, z = forward), so: optical_x = native_x,
        # optical_y = -native_z, optical_z = native_y. If frame_id is set to
        # a frame with the radar-native convention, this remap is wrong.
        buffer = []
        for p in points:
            buffer.append(struct.pack('ffff', p['x'], -p['z'], p['y'], p['intensity']))

        msg.data = b''.join(buffer)

        self.pointcloud_pub.publish(msg)

    def _cluster_points(self, positions):
        """Read this node's clustering parameters and call
        core/cluster.py's cluster_points_kdtree()."""
        return cluster_points_kdtree(
            positions,
            float(self.get_parameter('cluster_tolerance_m').value),
            int(self.get_parameter('cluster_min_points').value),
            int(self.get_parameter('cluster_max_points').value),
        )

    def publish_cluster_markers_msg(self, points, stamp):
        marker_array = MarkerArray()

        # Clear last frame's markers first. The number and ids of clusters
        # change every frame (live detections, not tracked objects), so
        # DELETEALL and republish is simpler than reconciling marker ids.
        clear_marker = Marker()
        clear_marker.header.frame_id = self.get_parameter('frame_id').value
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if points:
            # Same remap as publish_pointcloud(), so markers land on the
            # points they highlight.
            positions = np.array(
                [[p['x'], -p['z'], p['y']] for p in points], dtype=np.float64)
            intensities = np.array([p['intensity'] for p in points])

            clusters = self._cluster_points(positions)
            frame_id = self.get_parameter('frame_id').value
            lifetime = Duration(seconds=0.5).to_msg()

            for cluster_id, idxs in enumerate(clusters):
                cluster_pos = positions[idxs]
                cluster_intensity = intensities[idxs]
                centroid = cluster_pos.mean(axis=0)
                # Marker size follows the cluster's spatial extent, with a
                # visible minimum.
                spread = cluster_pos.max(axis=0) - cluster_pos.min(axis=0)
                diameter = float(np.clip(np.linalg.norm(spread), 0.08, 1.0))

                sphere = Marker()
                sphere.header.frame_id = frame_id
                sphere.header.stamp = stamp
                sphere.ns = 'radar_clusters'
                sphere.id = cluster_id
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = float(centroid[0])
                sphere.pose.position.y = float(centroid[1])
                sphere.pose.position.z = float(centroid[2])
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = diameter
                sphere.scale.y = diameter
                sphere.scale.z = diameter
                # Colour scaled by the cluster's mean intensity: redder means
                # a stronger return.
                mean_intensity = float(cluster_intensity.mean())
                strength = float(np.clip((mean_intensity - 4.0) / 4.0, 0.15, 1.0))
                sphere.color.r = 1.0
                sphere.color.g = 1.0 - strength
                sphere.color.b = 0.0
                sphere.color.a = 0.85
                sphere.lifetime = lifetime
                marker_array.markers.append(sphere)

                label = Marker()
                label.header.frame_id = frame_id
                label.header.stamp = stamp
                label.ns = 'radar_cluster_labels'
                label.id = cluster_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(centroid[0])
                label.pose.position.y = float(centroid[1])
                label.pose.position.z = float(centroid[2]) + diameter / 2 + 0.05
                label.pose.orientation.w = 1.0
                label.scale.z = 0.08
                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 0.9
                label.lifetime = lifetime
                label.text = f'{len(idxs)}pt {mean_intensity:.1f}'
                marker_array.markers.append(label)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = RadarPointCloudNode()

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
