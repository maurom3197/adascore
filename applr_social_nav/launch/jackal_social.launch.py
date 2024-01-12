from os import path
from os import environ
from os import pathsep
import yaml
import json
from scripts import GazeboRosPaths
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, SetEnvironmentVariable,
                            DeclareLaunchArgument, ExecuteProcess, Shutdown,
                            RegisterEventHandler, TimerAction, LogInfo,
                            GroupAction, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (PathJoinSubstitution, LaunchConfiguration,
                                  PythonExpression, EnvironmentVariable,
                                  FindExecutable)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                   OnProcessIO, OnProcessStart, OnShutdown)

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    # World generation parameters
    world_file_name = LaunchConfiguration('base_world')
    gz_obs = LaunchConfiguration('use_gazebo_obs')
    rate = LaunchConfiguration('update_rate')
    robot_name = LaunchConfiguration('robot_name')
    global_frame = LaunchConfiguration('global_frame_to_publish')
    use_navgoal = LaunchConfiguration('use_navgoal_to_start')
    ignore_models = LaunchConfiguration('ignore_models')
    use_gazebo_controllers = LaunchConfiguration('use_gazebo_controllers')
    use_localization = LaunchConfiguration('use_localization')
    this_package_dir = get_package_share_directory('applr_social_nav')
   
    # agent configuration file
    agent_conf_file = PathJoinSubstitution([
        this_package_dir,
        'config','agents_envs',
        LaunchConfiguration('configuration_file')
    ])

    configFilepath = path.join(
        this_package_dir, 'config','pic4rl_params',
        'main_params.yaml'
        )
    
    with open(configFilepath, 'r') as file:
        configParams = yaml.safe_load(file)['main_node']['ros__parameters']

    goals_path = path.join(this_package_dir, 
        'goals_and_poses', configParams['data_path'])
    goal_and_poses = json.load(open(goals_path,'r'))
    robot_pose, goal_pose = goal_and_poses["initial_pose"], goal_and_poses["goals"][0]

    x_rob = '-x '+str(robot_pose[0])
    y_rob = '-y '+str(robot_pose[1])
    z_rob = '-z '+str(0.07)
    yaw_rob = '-Y '+str(robot_pose[2])

    x_goal = '-x '+str(goal_pose[0])
    y_goal = '-y '+str(goal_pose[1])
    z_goal = '-z '+str(0.01)

    # Read the yaml file and load the parameters
    hunav_loader_node = Node(
        package='hunav_agent_manager',
        executable='hunav_loader',
        output='screen',
        parameters=[agent_conf_file]
    )

    # world base file
    world_file = path.join(this_package_dir, 
        'worlds', configParams["world_name"])
    # if desired to spawn goal model in Gazebo
    goal_entity = path.join(get_package_share_directory("gazebo_sim"), 'models', 
                'goal_box', 'model.sdf')

    # the node looks for the base_world file in the directory 'worlds'
    # of the package hunav_gazebo_plugin direclty. So we do not need to
    # indicate the path
    hunav_gazebo_worldgen_node = Node(
        package='hunav_gazebo_wrapper',
        executable='hunav_gazebo_world_generator',
        output='screen',
        parameters=[{'base_world': world_file},
                    {'use_gazebo_obs': gz_obs},
                    {'update_rate': rate},
                    {'robot_name': robot_name},
                    {'global_frame_to_publish': global_frame},
                    {'use_navgoal_to_start': use_navgoal},
                    {'ignore_models': ignore_models}]
    )

    ordered_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_loader_node,
            on_start=[
                LogInfo(
                    msg='HunNavLoader started, launching HuNav_Gazebo_world_generator after 2 seconds...'),
                TimerAction(
                    period=5.0,
                    actions=[hunav_gazebo_worldgen_node],
                )
            ]
        )
    )

    # Then, launch the generated world in Gazebo
    my_gazebo_models = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_wrapper'),
        'models',
    ])

    jackal_gazebo_models = str(Path(get_package_share_directory('jackal_description')).
                               parent.resolve())


    config_file_name = 'params.yaml'
    pkg_dir = get_package_share_directory('hunav_gazebo_wrapper')
    config_file = path.join(pkg_dir, 'launch', config_file_name)

    # the world generator will create this world
    # in this path
    world_path = PathJoinSubstitution([
        FindPackageShare('gazebo_sim'),
        'worlds',
        'generatedWorld.world'
    ])

    # Gazebo server
    gzserver_cmd = [
        'gzserver ',
        world_path,
        _boolean_command('verbose'), '',
        '-s ', 'libgazebo_ros_init.so',
        '-s ', 'libgazebo_ros_factory.so',
        # '-s ', 'libgazebo_ros_state.so',
        '--ros-args',
        '--params-file', config_file,
    ]

    # Gazebo client
    gzclient_cmd = [
        'gzclient',
        _boolean_command('verbose'), ' ',
    ]

    gzserver_process = ExecuteProcess(
        cmd=gzserver_cmd,
        output='screen',
        shell=True,
        on_exit=Shutdown(),
    )

    gzclient_process = ExecuteProcess(
        cmd=gzclient_cmd,
        output='screen',
        shell=True,
        on_exit=Shutdown(),
    )

    # Do not launch Gazebo until the world has been generated
    ordered_launch_event2 = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_gazebo_worldgen_node,
            on_start=[
                LogInfo(
                    msg='GenerateWorld started, launching Gazebo after 2 seconds...'),
                TimerAction(
                    period=2.0,
                    actions=[gzserver_process, gzclient_process],
                    # actions=[gzserver_process],
                )
            ]
        )
    )

    # jackal_controllers

    config_jackal_velocity_controller = PathJoinSubstitution(
        [FindPackageShare('jackal_gazebo'), 'config', 'control.yaml']
    )

    config_jackal_localization = PathJoinSubstitution(
        [FindPackageShare('jackal_gazebo'), 'config', 'localization.yaml']
    )

    config_twist_mux = PathJoinSubstitution(
        [FindPackageShare('jackal_gazebo'), 'config', 'twist_mux.yaml']
    )

    # Get URDF via xacro
    robot_description_command = [
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution(
            [FindPackageShare('jackal_description'),
             'urdf', 'jackal.urdf.xacro']
        ),
        ' ',
        'use_gazebo_controllers:=',
        use_gazebo_controllers,
        ' ',
        'gazebo_sim:=True',
        ' ',
        'gazebo_controllers:=',
        config_jackal_velocity_controller,
    ]

    launch_jackal_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('jackal_description'),
                 'launch',
                 'description.launch.py']
            )
        ),
        launch_arguments=[('robot_description_command',
                           robot_description_command)]
    )

    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_jackal',
        arguments=['-entity',
                   robot_name,
                   '-topic',
                   'robot_description',
                   x_rob, y_rob, z_rob, yaw_rob,],
        output='screen',
    )

    # Launch jackal_control/control.launch.py
    launch_jackal_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('jackal_control'), 'launch', 'control.launch.py']
        )),
        launch_arguments=[('robot_description_command', robot_description_command),
                          ('gazebo_sim', 'True'),
                          ('config_jackal_velocity',
                           config_jackal_velocity_controller),
                          ('config_jackal_localization',
                           config_jackal_localization),
                          ],
        condition = IfCondition(use_localization)
    )

    spawn_jackal_controllers = GroupAction([
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['jackal_velocity_controller',
                       '-c', '/controller_manager'],
            output='screen',
            condition=IfCondition(use_gazebo_controllers)
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
            output='screen',
            condition=IfCondition(use_gazebo_controllers)
        )
    ])

    # Stop the robot

    gzserver_cmd = [
        'gzserver ',
        world_path,
        _boolean_command('verbose'), '',
        '-s ', 'libgazebo_ros_init.so',
        '-s ', 'libgazebo_ros_factory.so',
        '--ros-args',
        '--params-file', config_file,
    ]

    stop_jackal_cmd = ['ros2 topic pub /stop/cmd_vel geometry_msgs/msg/Twist ',
                       '"{ linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"', ]

    stop_jackal = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[ExecuteProcess(
                cmd=stop_jackal_cmd,
                output='log',
                shell=True,
                on_exit=Shutdown(),
                condition=UnlessCondition(
                    use_gazebo_controllers)
            )],
        )
    )

    # Make sure spawn_jackal_controllers starts after spawn_robot
    jackal_controllers_spawn_callback = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[spawn_jackal_controllers],
        )
    )

    # Launch jackal_control/teleop_base.launch.py which is various ways to tele-op
    # the robot but does not include the joystick. Also, has a twist mux.
    launch_jackal_teleop_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('jackal_control'), 'launch', 'teleop_base.launch.py'])),
        launch_arguments=[('config_twist_mux', config_twist_mux)]
    )

    # hunav_manager node
    hunav_manager_node = Node(
        package='hunav_agent_manager',
        executable='hunav_agent_manager',
        name='hunav_agent_manager',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    static_tf_node = Node(
        package = "tf2_ros", 
        executable = "static_transform_publisher",
        output='screen',
        arguments = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch',
                        '0', '--roll', '0',
                          '--frame-id', 'map', '--child-frame-id', 'odom']
    )

    declare_agents_conf_file = DeclareLaunchArgument(
        'configuration_file', default_value='social_scenarios_agents.yaml',
        description='Specify configuration file name in the cofig directory'
    )

    declare_metrics_conf_file = DeclareLaunchArgument(
        'metrics_file', default_value='metrics.yaml',
        description='Specify the name of the metrics configuration file in the cofig directory'
    )

    declare_arg_world = DeclareLaunchArgument(
        'base_world', default_value='social_scenarios.world',
        description='Specify world file name'
    )
    declare_gz_obs = DeclareLaunchArgument(
        'use_gazebo_obs', default_value='true',
        description='Whether to fill the agents obstacles with closest Gazebo obstacle or not'
    )
    declare_update_rate = DeclareLaunchArgument(
        'update_rate', default_value='100.0',
        description='Update rate of the plugin'
    )
    declare_robot_name = DeclareLaunchArgument(
        'robot_name', default_value='jackal',
        description='Specify the name of the robot Gazebo model'
    )
    declare_frame_to_publish = DeclareLaunchArgument(
        'global_frame_to_publish', default_value='map',
        description='Name of the global frame in which the position of the agents are provided'
    )
    declare_use_navgoal = DeclareLaunchArgument(
        'use_navgoal_to_start', default_value='false',
        description='Whether to start the agents movements when a navigation goal is received or not'
    )
    declare_ignore_models = DeclareLaunchArgument(
        'ignore_models', default_value='ground_plane cafe',
        description='list of Gazebo models that the agents should ignore as obstacles as the ground_plane. Indicate the models with a blank space between them'
    )
    declare_arg_verbose = DeclareLaunchArgument(
        'verbose', default_value='false',
        description='Set "true" to increase messages written to terminal.'
    )

    declare_use_gazebo_controllers = DeclareLaunchArgument(
        'use_gazebo_controllers', default_value='true',
        description='Whether to start the gazebo controllers'
    )
    declare_use_localization = DeclareLaunchArgument(
        'use_localization', default_value='false',
        description='Whether to start the gazebo localization'
    )

    ld = LaunchDescription()

    # Declare the launch arguments
    ld.add_action(declare_agents_conf_file)
    ld.add_action(declare_metrics_conf_file)
    ld.add_action(declare_arg_world)
    ld.add_action(declare_gz_obs)
    ld.add_action(declare_update_rate)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_frame_to_publish)
    ld.add_action(declare_use_navgoal)
    ld.add_action(declare_ignore_models)
    ld.add_action(declare_arg_verbose)
    ld.add_action(declare_use_gazebo_controllers)
    ld.add_action(declare_use_localization)

    # Generate the world with the agents
    # launch hunav_loader and the WorldGenerator
    # 2 seconds later
    ld.add_action(hunav_loader_node)
    ld.add_action(ordered_launch_event)

    # hunav behavior manager node
    ld.add_action(hunav_manager_node)
    # hunav evaluator
    # ld.add_action(hunav_evaluator_node)

    # launch Gazebo after worldGenerator
    # (wait a bit for the world generation)
    # ld.add_action(gzserver_process)
    ld.add_action(ordered_launch_event2)
    # ld.add_action(gzclient_process)

    # launch jackal_description
    ld.add_action(launch_jackal_description)
    # spawn robot in Gazebo
    ld.add_action(spawn_robot)
    # launch jackal_control
    ld.add_action(launch_jackal_control)
    ld.add_action(jackal_controllers_spawn_callback)
    # launch jackal_teleop_base
    ld.add_action(launch_jackal_teleop_base)
    ld.add_action(static_tf_node)
    # stop the robot when no cmd_vel is received
    ld.add_action(stop_jackal)

    # ld.add_action(OpaqueFunction(function=printenv))

    return ld


# Add boolean commands if true
def _boolean_command(arg):
    cmd = ['"--', arg, '" if "true" == "',
           LaunchConfiguration(arg), '" else ""']
    py_cmd = PythonExpression(cmd)
    return py_cmd


def printenv(context, *args, **kwargs):
    import os
    print('env:', os.environ)
