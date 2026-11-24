#!/usr/bin/env python3
"""Bring up raw capture plus one or both processing pipelines.

    ros2 launch radar_pipelines radar_pipelines.launch.py
    ros2 launch radar_pipelines radar_pipelines.launch.py pipeline:=pointcloud
    ros2 launch radar_pipelines radar_pipelines.launch.py pipeline:=both cfar_method:=os
    ros2 launch radar_pipelines radar_pipelines.launch.py chirp_profile:=test

One argument picks the chirp profile for both halves
----------------------------------------------------
`chirp_profile` selects the .cfg handed to the driver and the calibration
sidecar the pipelines load, so the two cannot be set to different profiles
from here. The range calibration belongs to the chirp profile, and setting it
in a separate place lets the two drift apart.

If you launch the driver separately (start_driver:=false), set the nodes'
own `chirp_profile` parameter to match. The runtime mismatch guard
(core/calib.py's check_against_frame) logs an error on the first frame if it
does not.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution,
)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    xwr_share = get_package_share_directory('xwr_raw_ros')

    profile = LaunchConfiguration('chirp_profile')
    pipeline = LaunchConfiguration('pipeline')

    args = [
        DeclareLaunchArgument(
            'chirp_profile', default_value='near_mimo',
            description='Chirp profile name. Selects BOTH configs/6843aop/<name>.cfg '
                        'for the driver and calib/<name>.yaml for the pipelines.'),
        DeclareLaunchArgument(
            'pipeline', default_value='heatmap',
            description="Which pipeline(s) to run: 'heatmap', 'pointcloud', or 'both'."),
        DeclareLaunchArgument(
            'start_driver', default_value='true',
            description='Set false to attach the pipelines to an already-running driver '
                        '(or to a bag replay).'),
        DeclareLaunchArgument('radar_topic', default_value='radar1/radar_data'),
        # -- heatmap pipeline --
        DeclareLaunchArgument('azimuth_fov', default_value='60.0'),
        DeclareLaunchArgument('elevation_fov', default_value='45.0'),
        DeclareLaunchArgument(
            'doppler_window_bins', default_value='3',
            description='Half-width, in bins, of the Doppler window around zero velocity that '
                        'the heatmaps sum over (rather than the whole Doppler axis).'),
        # -- pointcloud pipeline --
        DeclareLaunchArgument(
            'cfar_method', default_value='ca',
            description="'ca' | 'os' | 'go' | 'so'. Try 'os' for multiple tabletop objects "
                        "inside one azimuth beamwidth -- see core/cfar.py."),
        DeclareLaunchArgument('cfar_threshold', default_value='2.5'),
        DeclareLaunchArgument('capon_angle_range', default_value='70'),
    ]

    radar_cfg = PathJoinSubstitution([
        TextSubstitution(text=os.path.join(xwr_share, 'configs', '6843aop')),
        [profile, TextSubstitution(text='.cfg')],
    ])

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(xwr_share, 'launch', 'radar.launch.py')),
        launch_arguments={'radar1_cfg': radar_cfg}.items(),
        condition=IfCondition(LaunchConfiguration('start_driver')),
    )

    run_heatmap = IfCondition(PythonExpression(
        ["'", pipeline, "' in ('heatmap', 'both')"]))
    run_pointcloud = IfCondition(PythonExpression(
        ["'", pipeline, "' in ('pointcloud', 'both')"]))

    heatmap_node = Node(
        package='radar_pipelines',
        executable='heatmap_node.py',
        name='radar_heatmap_rexo',
        output='screen',
        condition=run_heatmap,
        parameters=[{
            'chirp_profile': profile,
            'radar_topic': LaunchConfiguration('radar_topic'),
            'azimuth_fov': LaunchConfiguration('azimuth_fov'),
            'elevation_fov': LaunchConfiguration('elevation_fov'),
            'n_range_bins': 256,
            'n_azimuth_bins': 128,
            'n_elevation_bins': 128,
            'window_sidelobe_db': 65.0,
            'doppler_window_bins': LaunchConfiguration('doppler_window_bins'),
            'flip_aop_phase': True,
            'output_cartesian': True,
            'enable_visualization': True,
            'use_sim_time': False,
        }],
    )

    pointcloud_node = Node(
        package='radar_pipelines',
        executable='pointcloud_node.py',
        name='radar_pointcloud',
        output='screen',
        condition=run_pointcloud,
        parameters=[{
            'chirp_profile': profile,
            'radar_topic': LaunchConfiguration('radar_topic'),
            'cfar_method': LaunchConfiguration('cfar_method'),
            'cfar_threshold': LaunchConfiguration('cfar_threshold'),
            'capon_angle_range': LaunchConfiguration('capon_angle_range'),
            'use_sim_time': False,
        }],
    )

    return LaunchDescription(args + [driver, heatmap_node, pointcloud_node])
