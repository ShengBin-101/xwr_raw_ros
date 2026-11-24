#!/usr/bin/env python3
# Modified by Chan Sheng Bin, 2026: ported from ROS 1 (rospy) to ROS 2 (rclpy), with receiving and publishing in
#   separate threads. See NOTICE.

"""ROS 2 node: configure the radar and the DCA1000, and publish every frame's raw ADC data.

Publishes xwr_raw_ros/RadarFrameFull on `radar_data`, with the chirp parameters parsed from
the .cfg profile. One thread receives the DCA1000's UDP packets, another builds and
publishes the messages, so slow publishing drops whole frames rather than packets.
"""
import os
import sys
import argparse
import threading
from queue import Queue, Empty, Full

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from xwr_raw_ros.msg import RadarFrameFull
from xwr_raw.radar_config import RadarConfig
from xwr_raw.radar_pub import RadarPub


class RadarNode(Node):
    def __init__(self, args):
        super().__init__('xwr_radar')

        # Initialize publisher
        self.publisher = self.create_publisher(RadarFrameFull, 'radar_data', 1)

        # Parse and publish config file
        pkg_dir = get_package_share_directory('xwr_raw_ros')
        with open(os.path.join(pkg_dir, args.cfg), 'r') as f:
            cfg = f.readlines()

        radar_config = RadarConfig(cfg)

        # Extract params from config
        self.radar_params = radar_config.get_params()

        # Declare parameters
        for key, value in self.radar_params.items():
            self.declare_parameter(f'radar_params.{key}', value)

        # Configure and start radar capture
        self.radar = RadarPub(cfg,
                         cmd_tty        = args.cmd_tty,
                         dca_ip         = args.dca_ip,
                         dca_cmd_port   = int(args.dca_cmd_port),
                         host_ip        = args.host_ip,
                         host_cmd_port  = int(args.host_cmd_port),
                         host_data_port = int(args.host_data_port))
        self.radar.configure()
        self.radar.start_capture()

        # Pre-create message template with static fields (avoid recreating every frame)
        self._msg_template = RadarFrameFull()
        self._msg_template.header.frame_id = 'radar'
        self._msg_template.platform       = self.radar_params['platform']
        self._msg_template.adc_output_fmt = self.radar_params['adc_output_fmt']
        self._msg_template.range_bias     = self.radar_params['range_bias']
        self._msg_template.rx_phase_bias  = self.radar_params['rx_phase_bias']
        self._msg_template.chirp_time     = self.radar_params['chirp_time']
        self._msg_template.chirp_slope    = self.radar_params['chirp_slope']
        self._msg_template.frame_time     = self.radar_params['frame_time']
        self._msg_template.velocity_max   = self.radar_params['velocity_max']
        self._msg_template.velocity_res   = self.radar_params['velocity_res']
        self._msg_template.sample_rate    = self.radar_params['sample_rate']
        self._msg_template.range_max      = self.radar_params['range_max']
        self._msg_template.range_res      = self.radar_params['range_res']
        self._msg_template.rx = self.radar_params['rx']
        self._msg_template.tx = self.radar_params['tx']
        self._msg_template.shape = [self.radar_params['n_chirps'],
                                     self.radar_params['n_rx'],
                                     self.radar_params['n_samples']]

        # Frame queue - decouple fast packet reception from slow message building
        # Using maxsize=1 minimizes memory (only ~750KB overhead vs 3.75MB with maxsize=5)
        # If publish thread is slow, new frames get dropped (better than dropping packets)
        self._frame_queue = Queue(maxsize=1)

        # Start TWO threads: one for fast packet RX, one for message publishing
        self._running = True
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True, name="RadarRX")
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True, name="RadarPub")
        self._rx_thread.start()
        self._pub_thread.start()

        self.get_logger().info('Radar node initialized')

    def _receive_loop(self):
        """FAST tight loop - only receives UDP packets and updates frame buffer.
        Must be as fast as possible to prevent packet drops."""
        while self._running and rclpy.ok():
            try:
                # This is the tight loop - must be FAST
                frame_data, new_frame = self.radar.update_frame_buffer()
                if new_frame:
                    # Make a copy to avoid race conditions (frame_buffer reuses memory)
                    frame_copy = frame_data.copy()
                    # Non-blocking put - drop frame if queue full (better than dropping packets)
                    try:
                        self._frame_queue.put_nowait((self.get_clock().now().to_msg(), frame_copy))
                    except Full:
                        pass  # Queue full - drop this frame to keep receiving packets
            except Exception as e:
                self.get_logger().error(f'Error in RX loop: {e}')
                break

    def _publish_loop(self):
        """Slower loop - builds and publishes ROS messages.
        Decoupled from packet reception to prevent drops."""
        while self._running and rclpy.ok():
            try:
                # Block waiting for frame (this thread can be slow, won't affect packet RX)
                timestamp, frame_data = self._frame_queue.get(timeout=1.0)

                # Create message (copy template for static fields)
                msg = RadarFrameFull()
                msg.header.stamp = timestamp
                msg.header.frame_id = self._msg_template.header.frame_id

                # Copy static fields from template
                msg.platform       = self._msg_template.platform
                msg.adc_output_fmt = self._msg_template.adc_output_fmt
                msg.range_bias     = self._msg_template.range_bias
                msg.rx_phase_bias  = self._msg_template.rx_phase_bias
                msg.chirp_time     = self._msg_template.chirp_time
                msg.chirp_slope    = self._msg_template.chirp_slope
                msg.frame_time     = self._msg_template.frame_time
                msg.velocity_max   = self._msg_template.velocity_max
                msg.velocity_res   = self._msg_template.velocity_res
                msg.sample_rate    = self._msg_template.sample_rate
                msg.range_max      = self._msg_template.range_max
                msg.range_res      = self._msg_template.range_res
                msg.rx    = self._msg_template.rx
                msg.tx    = self._msg_template.tx
                msg.shape = self._msg_template.shape

                # Slow .tolist() conversion - but happens in separate thread!
                # RX thread keeps receiving packets during this time
                msg.data = frame_data.tolist()

                self.publisher.publish(msg)
            except Empty:
                # Timeout is normal when queue is empty
                pass
            except Exception as e:
                self.get_logger().error(f'Error in publish loop: {e}')
                break

    def shutdown(self):
        self._running = False
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        if self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)
        self.radar.close()


def main(args=None):
    rclpy.init(args=args)
    
    # Read path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help="Path to configuration file for radar.")
    parser.add_argument('--cmd_tty',        default='/dev/ttyUSB0',   help='CMD TTY of radar.')
    parser.add_argument('--dca_ip',         default='192.168.33.180', help='IP address of DCA1000.')
    parser.add_argument('--dca_cmd_port',   default=4096,             help='CMD port of DCA1000.')
    parser.add_argument('--host_ip',        default='192.168.33.30',  help='IP address of host.')
    parser.add_argument('--host_cmd_port',  default=4096,             help='CMD port of host.')
    parser.add_argument('--host_data_port', default=4098,             help='Data port of host.')
    parsed_args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    
    node = RadarNode(parsed_args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

