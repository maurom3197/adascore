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
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

	pic4slam		 	= get_package_share_directory('pic4slam')
	
	slam_toolbox = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(
			os.path.join(pic4slam, 'slam.launch.py')
			)
		)
	
	foot_link_tf = Node(
		package 	= 'tf2_ros',
		executable 	= 'static_transform_publisher',
		arguments 	= ["0", "0", "0.05", "0", "0", "0", "base_footprint", "base_link"]
		)
	
	robot_laser_tf = Node(
		package 	= 'tf2_ros',
		executable 	= 'static_transform_publisher',
		arguments 	= ["0.07", "0", "0", "0", "0", "0", "base_link", "laser_frame"]
		)
	
	return LaunchDescription([
		slam_toolbox,
		foot_link_tf,
		robot_laser_tf
		])

