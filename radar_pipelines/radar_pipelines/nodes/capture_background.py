#!/usr/bin/env python3

"""Trigger a static-background capture on a running pipeline node, and wait.

    ros2 run radar_pipelines capture_background.py --node /radar_pointcloud
    ros2 run radar_pipelines capture_background.py --node /radar_heatmap_rexo

Run with the scene physically cleared of objects (mount, table and
backdrop only). The node averages `background_n_frames` frames at the stage
where the subtraction is later applied, saves the baseline to its
background_dir, and resets its own trigger. This script sets the trigger,
waits for the reset and reports.

The capture happens inside the pipeline, not in this script. The published
heatmap topics come after condition_heatmap(), while the subtraction is
applied before it, so a baseline recorded from a topic would be one
processing stage off and would not line up bin for bin (see
core/clutter.py).
"""

import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class BackgroundTrigger(Node):
    def __init__(self, target):
        super().__init__('radar_background_trigger')
        self.target = target.rstrip('/')
        self.set_cli = self.create_client(SetParameters, f'{self.target}/set_parameters')
        self.get_cli = self.create_client(GetParameters, f'{self.target}/get_parameters')

    def wait_for_services(self, timeout_sec):
        deadline = time.time() + timeout_sec
        for cli, name in ((self.set_cli, 'set_parameters'), (self.get_cli, 'get_parameters')):
            while not cli.wait_for_service(timeout_sec=0.5):
                if time.time() > deadline:
                    self.get_logger().error(
                        f'No {self.target}/{name} service after {timeout_sec:.0f}s -- '
                        f'is the node running, and is the name right? '
                        f'Check with: ros2 node list')
                    return False
        return True

    def _call(self, client, request, timeout_sec=5.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        return future.result()

    def get_bool(self, name):
        req = GetParameters.Request()
        req.names = [name]
        res = self._call(self.get_cli, req)
        if res is None or not res.values:
            return None
        if res.values[0].type != ParameterType.PARAMETER_BOOL:
            return None
        return res.values[0].bool_value

    def get_int(self, name):
        req = GetParameters.Request()
        req.names = [name]
        res = self._call(self.get_cli, req)
        if res is None or not res.values:
            return None
        return res.values[0].integer_value

    def set_bool(self, name, value):
        req = SetParameters.Request()
        p = Parameter()
        p.name = name
        p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(value))
        req.parameters = [p]
        res = self._call(self.set_cli, req)
        return bool(res and res.results and res.results[0].successful)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default='/radar_pointcloud',
                        help='Fully-qualified node name to trigger '
                             '(e.g. /radar_pointcloud or /radar_heatmap_rexo).')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Seconds to wait for the capture to complete.')
    parsed, _ = parser.parse_known_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=args)
    node = BackgroundTrigger(parsed.node)
    exit_code = 0
    try:
        if not node.wait_for_services(10.0):
            return 1

        n_frames = node.get_int('background_n_frames')
        node.get_logger().info(
            f'Triggering background capture on {parsed.node} '
            f'({n_frames if n_frames else "?"} frames). '
            f'Make sure the scene is physically CLEAR of objects.')

        if not node.set_bool('background_capture', True):
            node.get_logger().error('Failed to set background_capture -- does that node have it?')
            return 1

        # The node resets the trigger itself once the capture completes, so
        # the trigger flipping back to false is the completion signal. There
        # is no capture-done topic, so this polls the parameter.
        deadline = time.time() + parsed.timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.get_bool('background_capture') is False:
                node.get_logger().info(
                    'Background captured and saved. Put the objects back; set '
                    'background_enabled:=true on the node if it is not already. '
                    'The baseline is reloaded automatically on the next restart.')
                break
        else:
            node.get_logger().error(
                f'Timed out after {parsed.timeout:.0f}s. Is the radar publishing? '
                f'The node needs {n_frames if n_frames else "N"} frames, which at 5Hz '
                f'is ~{(n_frames or 30) / 5:.0f}s.')
            node.set_bool('background_capture', False)
            exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main() or 0)
