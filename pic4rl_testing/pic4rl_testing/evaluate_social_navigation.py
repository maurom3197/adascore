#!/usr/bin/env python3

import os
import numpy as np
from numpy import savetxt
import math
import subprocess
import json
import random
import sys
import time
import yaml
import logging
import datetime
from pathlib import Path

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose,PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from pic4rl_testing.utils.generic_sensor import Sensors
from pic4rl_testing.utils.env_utils import *
from pic4rl_testing.utils.sfm import SocialForceModel

from rclpy.parameter import Parameter
#from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterValue

#from pic4rl.nav_param_client import DWBparamsClient
from nav2_simple_commander.robot_navigator import BasicNavigator, NavigationResult
from pic4rl_testing.utils.nav_metrics import Navigation_Metrics
from people_msgs.msg import People

class EvaluateSocialNav(Node):
    def __init__(self):
        super().__init__('evaluate_social_nav')
        
        #rclpy.logging.set_logger_level('evaluate_social_nav', 10)
        goals_path      = os.path.join(
            get_package_share_directory('pic4rl_testing'), 'goals_and_poses')
        main_params_path  = os.path.join(
            get_package_share_directory('pic4rl_testing'), 'config', 'main_params.yaml')
        train_params_path= os.path.join(
            get_package_share_directory('pic4rl_testing'), 'config', 'training_params.yaml')
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), 
            'models/goal_box/model.sdf'
            )
        
        with open(main_params_path, 'r') as file:
            main_params = yaml.safe_load(file)['main_node']['ros__parameters']
        with open(train_params_path, 'r') as train_param_file:
            training_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(
            namespace   = '',
            parameters  = [
                ('data_path', main_params['data_path']),
                ('n_experiments', training_params['--n-experiments']),
                ('change_goal_and_pose', training_params['--change_goal_and_pose']),
                ('starting_episodes', training_params['--starting_episodes']),
                ('timeout_steps', training_params['--episode-max-steps']),
                ('robot_name', main_params['robot_name']),
                ('agents_config', main_params['agents_config']),
                ('goal_tolerance', main_params['goal_tolerance']),
                ('update_frequency', main_params['applr_param']['update_frequency']),
                ('lidar_dist', main_params['laser_param']['max_distance']),
                ('lidar_points', main_params['laser_param']['num_points']),
                ('gazebo_client', main_params['gazebo_client'])
                ]
            )

        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path      = os.path.join(goals_path, self.data_path)
        self.n_experiments = self.get_parameter(
            'n_experiments').get_parameter_value().integer_value
        self.change_episode = self.get_parameter(
            'change_goal_and_pose').get_parameter_value().integer_value
        self.starting_episodes = self.get_parameter(
            'starting_episodes').get_parameter_value().integer_value
        self.timeout_steps  = self.get_parameter(
            'timeout_steps').get_parameter_value().integer_value
        self.robot_name     = self.get_parameter(
            'robot_name').get_parameter_value().string_value
        self.agents_config     = self.get_parameter(
            'agents_config').get_parameter_value().string_value
        self.goal_tolerance     = self.get_parameter(
            'goal_tolerance').get_parameter_value().double_value
        self.params_update_freq   = self.get_parameter(
            'update_frequency').get_parameter_value().double_value
        self.lidar_distance = self.get_parameter(
            'lidar_dist').get_parameter_value().double_value
        self.lidar_points   = self.get_parameter(
            'lidar_points').get_parameter_value().integer_value
        self.gazebo_client = self.get_parameter(
            'gazebo_client').get_parameter_value().bool_value

        # create log dir 
        self.log_folder_name, self.logdir = create_logdir('nav2', 'social_plugin', training_params['--logdir'])

        self.create_clients()
        self.unpause()

        # create Sensor class to get and process sensor data
        self.sensors = Sensors(self)
        self.spin_sensors_callbacks()
        qos = QoSProfile(depth=10)

        # init goal publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)

        # self.get_logger().info('People topic subscription')
        # self.people_sub = self.create_subscription(
        #         People,
        #         '/people', 
        #         self.people_callback,
        #         1)

        self.sfm = SocialForceModel(self, self.agents_config)

        self.entity_path = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')

        self.episode_step = 0
        self.previous_twist = None
        self.previous_event = "None"
        self.prev_nav_state = "unknown"
        self.simulation_restarted = 0
        self.failure_counter = 0
        self.episode = 0
        self.collision_count = 0
        self.min_obstacle_distance = 4.0
        self.t0 = 0.0
        self.index = 0
        self.people_state = []
        self.k_people = 4
        self.min_people_distance = 10.0

        self.initial_pose, self.goals, self.poses, self.agents = get_goals_and_poses(self.data_path)
        self.start_pose = [0., 0., 0.]
        self.goal_pose = self.goals[0]
        self.init_nav_params = [0.25, 0.25, # covariance height/width
                                0.25, # covariance static
                                #   0.25, 0.25, # covariance right
                                #   0.6, 1.5 # max vel robot
                                ]

        self.n_navigation_end = 0
        self.navigator = BasicNavigator()
        self.nav_metrics = Navigation_Metrics(main_params, self.logdir)

        self.get_logger().info("PIC4RL_Environment: Starting process")

    def step(self, episode_step=0):
        """
        """
        self.get_logger().debug("Env step : " + str(episode_step))
        self.episode_step = episode_step

        done = self._step()

        return done


    def _step(self, nav_params=None, reset_step = False):
        """
        """
        self.spin_sensors_callbacks()
        self.get_logger().debug("getting sensor data...")
        lidar_measurements, goal_info, robot_pose, collision = self.get_sensor_data()

        self.get_logger().debug("getting people data...")
        people_state, people_info = self.get_people_state(robot_pose)

        self.get_logger().debug("checking events...")
        done, event = self.check_events(lidar_measurements, goal_info, robot_pose, collision)

        self.get_logger().debug("ccomputing social work...")
        wr, wp = self.sfm.computeSocialWork()
        Rs = wr + wp

        self.get_logger().debug("collecting metrics data...")
        self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step, costmap_params=nav_params, social_work=Rs, done=done)

        # Regulate the step frequency of the environment
        # action_hz, t1 = compute_frequency(self.t0)
        # self.t0 = t1
        # frequency_control(self.params_update_freq)
        # self.get_logger().debug('Sending action at '+str(action_hz))

        self.update_state(lidar_measurements, goal_info, robot_pose, people_state, nav_params, done, event)

        if done:
            # compute metrics at the end of the episode
            self.get_logger().debug("Computing episode metrics...")
            self.nav_metrics.calc_metrics(self.episode, self.start_pose, self.goal_pose)
            self.nav_metrics.log_metrics_results(self.episode)
            self.nav_metrics.save_metrics_results(self.episode)

        return done

    def spin_sensors_callbacks(self):
        """
        """
        self.get_logger().debug("spinning for sensor_msg...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def get_sensor_data(self):
        """
        """
        sensor_data = {}
        sensor_data["scan"], min_obstacle_distance, collision = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()

        if sensor_data["scan"] is None:
            self.get_logger().debug("scan data is None...")
            sensor_data["scan"] = np.squeeze(np.ones((1,self.lidar_points))*2.0).tolist()
            min_obstacle_distance = 2.0
            collision = False
        if sensor_data["odom"] is None:
            self.get_logger().debug("odom data is None...")
            sensor_data["odom"] = [0.0,0.0,0.0]

        goal_info, robot_pose = process_odom(self.goal_pose, sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]
        self.min_obstacle_distance = min_obstacle_distance

        return lidar_measurements, goal_info, robot_pose, collision

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """
        """
        # get action feedback from navigator
        feedback = self.navigator.getFeedback()
        #self.get_logger().debug('Navigator feedback: '+str(feedback))
        # check if navigation is complete
        if self.navigator.isNavComplete():
            result = check_navigation(self.navigator)
            if (result == NavigationResult.FAILED or result == NavigationResult.CANCELED):
                self.send_goal(self.goal_pose)
                self.n_navigation_end = self.n_navigation_end +1
                if self.n_navigation_end == 50:
                    self.get_logger().info('Navigation aborted more than 50 times... pausing Nav till next episode.') 
                    self.get_logger().info(f"Test Episode {self.episode+1}: Nav failed")
                    logging.info(f"Test Episode {self.episode+1}: Nav failed")
                    self.prev_nav_state = "nav2_failed"
                    return True, "nav2_failed"  
                
            if result == NavigationResult.SUCCEEDED:
                if self.prev_nav_state == "goal":
                    self.get_logger().info('uncorrect goal status detected... resending goal.') 
                    self.send_goal(self.goal_pose)
                    return False, "None"

                self.get_logger().info(f"Test Episode {self.episode+1}: Goal")
                logging.info(f"Test Episode {self.episode+1}: Goal")
                self.prev_nav_state = "goal"
                return True, "goal"

        else:
            self.prev_nav_state = "unknown"

        # check collision
        if  collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(f"Test Episode {self.episode+1}: Collision")
                logging.info(f"Test Episode {self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        # check timeout steps
        if self.episode_step == self.timeout_steps:
            self.get_logger().info(f"Test Episode {self.episode+1}: Timeout")
            logging.info(f"Test Episode {self.episode+1}: Timeout")
            return True, "timeout"

        return False, "None"


    def update_state(self,lidar_measurements, goal_info, robot_pose, people_state, nav_params, done, event):
        """
        """
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose
        self.people_state = people_state
        self.previous_nav_params = nav_params
        self.previous_event = event

    def reset(self, episode):
        """
        """
        self.episode = episode

        self.get_logger().debug("pausing...")
        self.pause()

        self.navigator.cancelNav()
        subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 1}'",
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(3.0)

        self.get_logger().info(f"Initializing new episode: scenario {self.index}")
        logging.info(f"Initializing new episode: scenario {self.index}")
        self.new_episode()

        self.get_logger().debug("unpausing...")
        self.unpause()

        self.get_logger().debug("Performing null step to reset variables")

        _ = self._step(reset_step = True)
    
    def new_episode(self):
        """
        """
        self.get_logger().debug("Respawning agents ...")
        self.respawn_agents()
        
        self.get_logger().debug("Respawning robot ...")
        self.respawn_robot(self.index)
    
        self.get_logger().debug("Respawning goal ...")
        self.respawn_goal(self.index)

        self.get_logger().debug("Resetting navigator ...")
        self.reset_navigator(self.index)

        self.index = self.index+1 if self.index<len(self.poses)-1 else 0 

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):
        """
        """
        x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.get_logger().info(f"Test Episode {self.episode+1}, robot pose [x,y,yaw]: {self.start_pose}")
        logging.info(f"Test Episode {self.episode+1}, robot pose [x,y,yaw]: {self.start_pose}")

        position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(0.07)+"}"
        orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
        pose = position+", "+orientation
        state = "'{state: {name: '"+self.robot_name+"',pose: {"+pose+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(2.0)

    def respawn_agents(self,):
        """
        """
        # if self.index == 0:
        #     agents2reset = [0,1,2,3]
        # elif self.index == 2:
        #     agents2reset = [5]
        # elif self.index == 3:
        #     agents2reset = [6]
        # else:
        #     return

        if self.index <= 4:
            agents2reset = [1]
        else:
            agents2reset = [1,6,7,9,10,11]
            
        for agent in agents2reset:
            x, y , yaw = tuple(self.agents[agent-1])

            self.get_logger().info(f"Agent pose [x,y,yaw]: {[x, y, yaw]}")
            agent_name = "agent"+str(agent)

            position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(1.50)+"}"
            #orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
            #pose = position+", "+orientation
            state = "'{state: {name: '"+agent_name+"',pose: {"+position+"}}}'"
            subprocess.run(
                "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
                shell=True,
                stdout=subprocess.DEVNULL
                )
            time.sleep(1.0)

        #ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState '{state: {name: 'agent2',pose: {position: {x: 2.924233, y: 5.0, z: 1.25}}}}'

    def respawn_goal(self, index):
        """
        """
        self.get_goal(index)

        self.get_logger().info(f"Test Episode {self.episode+1}, goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Test Episode {self.episode+1}, goal pose [x, y]: {self.goal_pose}")

    def reset_navigator(self, index):
        self.get_logger().debug("Restarting LifeCycleNodes...")
        subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 2}'",
                shell=True,
                stdout=subprocess.DEVNULL
                )

        self.n_navigation_end = 0

        self.get_logger().debug("wait until Nav2Active...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().debug("Clearing all costmaps...")
        self.navigator.clearAllCostmaps()
        time.sleep(5.0)
 
        self.get_logger().debug("Sending goal ...")
        self.send_goal(self.goal_pose)

    def get_people_state(self, robot_pose):
        """
        """
        people_state, people_info, min_people_distance = self.sfm.get_people(robot_pose)
        self.min_people_distance = min_people_distance

        return people_state, people_info

    def get_goal(self, index):
        # get goal from predefined list
        self.goal_pose = self.goals[index]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose))

    def send_goal(self,pose):
        # Set the robot's goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'odom'
        goal_pose.pose.position.x = pose[0]
        goal_pose.pose.position.y = pose[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)
        self.navigator.goToPose(goal_pose)
        
    def pause(self):

        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pause_physics_client.call_async(req) 

    def unpause(self):

        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.unpause_physics_client.call_async(req)

    def create_clients(self,):
        # create reset world client 
        #self.reset_world_client = self.create_client(Empty, 'reset_simulation')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

    def evaluate(self,):
        """
        """
        self.unpause()
        for n in range(self.n_experiments):
            for episode in range(len(self.goals)):
                episode_steps = 0
                self.reset(episode)
                done = False

                while not done:
                    done = self.step(episode_steps)
                    episode_steps += 1
        self.pause()


def main(args=None):
    rclpy.init(args=args)

    evaluate_social_nav = EvaluateSocialNav()
    try:
        evaluate_social_nav.evaluate()
    except Exception as e:
            evaluate_social_nav.get_logger().error("Error in starting nav evaluation node: {}".format(e))
    
    evaluate_social_nav.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()