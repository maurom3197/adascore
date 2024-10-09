from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile, Parameter, ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
import yaml

from nav2_common.launch import ReplaceString, RewrittenYaml

def generate_launch_description():
    # Launch configuration variables specific to simulation
    pkg_name = LaunchConfiguration("pkg_name")
    task = LaunchConfiguration("task")
    sensor = LaunchConfiguration("sensor")

    # Declare the launch arguments

    sensor_arg = DeclareLaunchArgument(
        "sensor", default_value="", description="sensor type: camera or lidar, adascore"
    )

    task_arg = DeclareLaunchArgument(
        "task",
        default_value="",
        description="task type: goToPose, Following, Vineyards, SocialController",
    )

    pkg_name_arg = DeclareLaunchArgument(
        "pkg_name", default_value="adascore", description="package name"
    )

    start_pic4rl = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('pic4rl'),
        'launch', 'pic4rl_starter.launch.py')
      ),
      launch_arguments={
        # 'sensor': sensor_arg,
        'task': task,
        'pkg_name': pkg_name,
        'sensor': sensor,
        'main_params': PathJoinSubstitution([FindPackageShare(pkg_name), 'config','pic4rl_params', 'main_params.yaml']),
        'training_params': PathJoinSubstitution([FindPackageShare(pkg_name), 'config','pic4rl_params', 'training_params.yaml']),
      }.items()
    )

    # Specify the actions
    ld = LaunchDescription()
    ld.add_action(sensor_arg)
    ld.add_action(task_arg)
    ld.add_action(pkg_name_arg)
    ld.add_action(start_pic4rl)
    return ld
