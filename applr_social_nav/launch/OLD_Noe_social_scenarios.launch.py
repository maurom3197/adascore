"""
Copyright 2021 PIC4SeR All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================
Author: Andrea Eirale
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    # Get the launch directory
    pkg_path = get_package_share_directory("pic4nav")
    pic4nav_config = os.path.join(pkg_path, "config")
    params_file = os.path.join(pic4nav_config, "social_scenarios_nav_params.yaml")
    # default_nav_to_pose_bt_xml = os.path.join(
    #     pic4nav_config, "bt_recovery.xml"
    # )
    bt_tree_dir = get_package_share_directory("nav2_bt_navigator")
    bt_tree_dir = os.path.join(bt_tree_dir, "behavior_trees")
    default_nav_to_pose_bt_xml = os.path.join(
        bt_tree_dir, "navigate_to_pose_w_replanning_and_recovery.xml"
    )
    default_nav_through_poses_bt_xml = os.path.join(
        bt_tree_dir, "navigate_through_poses_w_replanning_and_recovery.xml"
    )

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    map_yaml_file = LaunchConfiguration("map")
    # params_file = LaunchConfiguration('params_file', params_file)
    map_subscribe_transient_local = LaunchConfiguration("map_subscribe_transient_local")
    use_amcl = LaunchConfiguration("use_amcl", default=False)

    lifecycle_nodes = [
        "map_server",
        # 'amcl',
        "controller_server",
        "waypoint_follower",
        "planner_server",
        "behavior_server",
        "bt_navigator",
    ]

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    # TODO(orduno) Substitute with `PushNodeRemapping`
    #              https://github.com/ros2/launch_ros/issues/56
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        "use_sim_time": use_sim_time,
        "default_nav_to_pose_bt_xml": default_nav_to_pose_bt_xml,
        "default_nav_through_poses_bt_xml": default_nav_through_poses_bt_xml,
        "autostart": autostart,
        "map_subscribe_transient_local": map_subscribe_transient_local,
        "yaml_filename": map_yaml_file,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True,
    )

    return LaunchDescription(
        [
            # Set env var to print messages to stdout immediately
            # SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
            DeclareLaunchArgument(
                "namespace", default_value="", description="Top-level namespace"
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="True",
                description="Use simulation (Gazebo) clock if true",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="True",
                description="Automatically startup the nav2 stack",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=params_file,
                description="Full path to the ROS2 parameters file to use",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(
                    pkg_path, "maps", "map_social_scenarios.yaml"
                ),
                description="Full path to map yaml file to load",
            ),
            DeclareLaunchArgument(
                "default_nav_to_pose_bt_xml",
                default_value=default_nav_to_pose_bt_xml,
                description="Full path to the behavior tree xml file to use",
            ),
            DeclareLaunchArgument(
                "default_nav_through_poses_bt_xml",
                default_value=default_nav_through_poses_bt_xml,
                description="Full path to the behavior tree xml file to use",
            ),
            DeclareLaunchArgument(
                "map_subscribe_transient_local",
                default_value="True",
                description="Whether to set the map subscriber QoS to transient local",
            ),
            # DeclareLaunchArgument(
            #     'use_amcl', default_value='False',
            #     description='Whether to use the amcl localization or setting an static tf'),
            # map server
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            # AMCL localization
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
                # condition=IfCondition(AndSubstitution(NotSubstitution(run_headless), use_rviz))
                condition=IfCondition(PythonExpression([use_amcl])),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
                # other option: arguments = "0 0 0 0 0 0 pmb2 base_footprint".split(' ') Unless
                condition=UnlessCondition(PythonExpression([use_amcl])),
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=[configured_params],
                remappings=remappings,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"autostart": autostart},
                    {"node_names": lifecycle_nodes},
                ],
            ),
        ]
    )
