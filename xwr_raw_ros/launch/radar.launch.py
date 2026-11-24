#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('xwr_raw_ros')
    
    # Declare launch arguments
    radar1_arg = DeclareLaunchArgument('radar1', default_value='radar1')
    radar1_cfg_arg = DeclareLaunchArgument('radar1_cfg', 
                                            default_value=os.path.join(pkg_dir, 'configs/6843aop/near.cfg'))
    radar1_cmd_tty_arg = DeclareLaunchArgument('radar1_cmd_tty', 
                                                default_value='/dev/ttyUSB0')
    radar1_dca_ip_arg = DeclareLaunchArgument('radar1_dca_ip', 
                                               default_value='192.168.33.180')
    radar1_dca_cmd_port_arg = DeclareLaunchArgument('radar1_dca_cmd_port', 
                                                     default_value='4096')
    radar1_host_ip_arg = DeclareLaunchArgument('radar1_host_ip', 
                                                default_value='192.168.33.30')
    radar1_host_cmd_port_arg = DeclareLaunchArgument('radar1_host_cmd_port', 
                                                      default_value='4096')
    radar1_host_data_port_arg = DeclareLaunchArgument('radar1_host_data_port', 
                                                       default_value='4098')
    
    # Get launch configurations
    radar1 = LaunchConfiguration('radar1')
    radar1_cfg = LaunchConfiguration('radar1_cfg')
    radar1_cmd_tty = LaunchConfiguration('radar1_cmd_tty')
    radar1_dca_ip = LaunchConfiguration('radar1_dca_ip')
    radar1_dca_cmd_port = LaunchConfiguration('radar1_dca_cmd_port')
    radar1_host_ip = LaunchConfiguration('radar1_host_ip')
    radar1_host_cmd_port = LaunchConfiguration('radar1_host_cmd_port')
    radar1_host_data_port = LaunchConfiguration('radar1_host_data_port')
    
    # Radar node
    radar_node = Node(
        package='xwr_raw_ros',
        executable='radar.py',
        name='xwr_radar',
        output='screen',
        arguments=[
            radar1_cfg,
            '--cmd_tty', radar1_cmd_tty,
            '--dca_ip', radar1_dca_ip,
            '--dca_cmd_port', radar1_dca_cmd_port,
            '--host_ip', radar1_host_ip,
            '--host_cmd_port', radar1_host_cmd_port,
            '--host_data_port', radar1_host_data_port
        ]
    )
    
    # Group nodes with namespace
    radar_group = GroupAction([
        PushRosNamespace(radar1),
        radar_node,
    ])
    
    return LaunchDescription([
        radar1_arg,
        radar1_cfg_arg,
        radar1_cmd_tty_arg,
        radar1_dca_ip_arg,
        radar1_dca_cmd_port_arg,
        radar1_host_ip_arg,
        radar1_host_cmd_port_arg,
        radar1_host_data_port_arg,
        radar_group
    ])
