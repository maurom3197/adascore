import os
from ament_index_python.packages import get_package_share_directory
import launch
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('pic4rl'),
        'config',
        'ros_params.yaml')

    gazebo_sim = get_package_share_directory('gazebo_sim')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_sim,'launch', 'simulation.launch.py')
            )
        )

    pic4rl_training = Node(
        package='pic4rl',
        node_executable='pic4rl_training_camera',
        node_name='pic4rl_training_camera',
        prefix=['stdbuf -o L'],
        output='screen',
        parameters=[config])
    
    return launch.LaunchDescription([
        sim,
        TimerAction(period=5., actions=[pic4rl_training])
    ])
