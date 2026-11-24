#!/usr/bin/env python3
"""Minimal live viewer for the heatmap node's dual Cartesian heatmaps.

Subscribes to radar_rexo/hm_hori_viz and radar_rexo/hm_vert_viz (colourized
bgr8 images from the heatmap node's publish_visualization(), enabled by its
enable_visualization parameter, default True) and shows them side by side
in one window. No detection or calibration logic: this only displays the
pipeline's output.

Run (after sourcing this workspace's install/setup.bash):
    ros2 run radar_pipelines view_heatmaps.py
Requires a display (X11) to show the window. Without one the window fails
to open.

Left panel: hm_hori (range-azimuth, bird's-eye view)
Right panel: hm_vert (range-elevation, side view)
Press 'q' in the window to quit.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class HeatmapViewer(Node):
    def __init__(self):
        super().__init__("radar_heatmap_viewer")
        self.bridge = CvBridge()
        self.latest_hori = None
        self.latest_vert = None
        self.n_hori = 0
        self.n_vert = 0
        self.create_subscription(Image, "radar_rexo/hm_hori_viz", self._on_hori,
                                  qos_profile_sensor_data)
        self.create_subscription(Image, "radar_rexo/hm_vert_viz", self._on_vert,
                                  qos_profile_sensor_data)
        self.get_logger().info(
            "Subscribed to radar_rexo/hm_hori_viz and radar_rexo/hm_vert_viz. "
            "Waiting for frames...")

    def _on_hori(self, msg):
        self.latest_hori = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.n_hori += 1

    def _on_vert(self, msg):
        self.latest_vert = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.n_vert += 1


def main():
    rclpy.init()
    node = HeatmapViewer()
    cv2.namedWindow("radar heatmaps: hori (range-azimuth) | vert (range-elevation)")

    printed_first_frame = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            if node.latest_hori is not None and node.latest_vert is not None:
                if not printed_first_frame:
                    node.get_logger().info(
                        f"First frames received (hori shape={node.latest_hori.shape}, "
                        f"vert shape={node.latest_vert.shape}) -- pipeline is live.")
                    printed_first_frame = True

                hori = node.latest_hori
                vert = node.latest_vert
                if hori.shape != vert.shape:
                    vert = cv2.resize(vert, (hori.shape[1], hori.shape[0]))
                combined = np.hstack([hori, vert])
                combined = cv2.putText(
                    combined.copy(), f"hori frames: {node.n_hori}  vert frames: {node.n_vert}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("radar heatmaps: hori (range-azimuth) | vert (range-elevation)",
                           combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
