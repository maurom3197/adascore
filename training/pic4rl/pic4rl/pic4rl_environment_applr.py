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
from pic4rl.generic_sensor import Sensors
from pic4rl.nav_param_client import DWBparamsClient
from nav2_simple_commander.robot_navigator import BasicNavigator, NavigationResult

class Pic4rlEnvironmentAPPLR(Node):
    def __init__(self, dwb_client):
        super().__init__('pic4rl_env_applr')
        
        #rclpy.logging.set_logger_level('pic4rl_env_applr', 10)
        goals_path = os.path.join(get_package_share_directory('pic4rl'), 'goals_and_poses')
        configFilepath = os.path.join(get_package_share_directory('pic4rl'), 'config', 'main_param.yaml')
        with open(configFilepath, 'r') as file:
            configParams = yaml.safe_load(file)['main_node']['ros__parameters']

        self.declare_parameters(namespace='',
        parameters=[
            ('data_path', configParams['data_path']),
            ('change_goal_and_pose', configParams['change_goal_and_pose']),
            ('timeout_steps', configParams['timeout_steps']),
            ('robot_name', configParams['robot_name']),
            ('lidar_points', configParams["laser_param"]["num_points"]),
            ])

        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.data_path = os.path.join(goals_path, self.data_path)
        self.change_episode = self.get_parameter('change_goal_and_pose').get_parameter_value().integer_value
        self.timeout_steps = self.get_parameter('timeout_steps').get_parameter_value().integer_value
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.lidar_points = self.get_parameter('lidar_points').get_parameter_value().integer_value


        # create Sensor class to get and process sensor data
        self.sensors = Sensors(self)
        qos = QoSProfile(depth=10)
        # init goal publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)

        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        self.entity_path = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')
        self.init_step = True
        self.episode_step = 0
        self.starting_episodes = 12
        self.previous_twist = None
        self.episode = 0
        self.collision_count = 0
        self.min_obstacle_distance = 2.0

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.init_dwb_params = [0.35, 1.0, 20, 20, 0.02, 32.0, 24.0, 0.55]
        self.n_navigation_aborted = 0
        self.dwb_client = dwb_client
        self.navigator = BasicNavigator()

        self.get_logger().info("PIC4RL_Environment: Starting process")

    def step(self, action):
        """
        """
        self.get_logger().debug("env step...")
        dwb_params = action.tolist()
        dwb_params[2] = int(dwb_params[2])
        dwb_params[3] = int(dwb_params[3])
        self.get_logger().debug("dwb_params: "+str(dwb_params))

        observation, reward, done = self._step(dwb_params)
        info = None

        return observation, reward, done, info

    def _step(self, dwb_params=None, reset_step = False):
        """
        """
        self.get_logger().debug("sending action...")

        self.send_action(dwb_params)

        self.get_logger().debug("getting sensor data...")
        lidar_measurements, goal_info, robot_pose, collision = self.get_sensor_data()

        self.get_logger().debug("checking events...")
        done, event = self.check_events(lidar_measurements, goal_info, robot_pose, collision)

        if not reset_step:
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(lidar_measurements, goal_info, robot_pose, done, event)

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(lidar_measurements, goal_info, robot_pose, dwb_params)

        else:
            reward = None
            observation = None

        self.update_state(lidar_measurements, goal_info, robot_pose, dwb_params, done, event)

        return observation, reward, done

    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path,'r'))

        return data["initial_pose"], data["goals"], data["poses"]

    def send_action(self,dwb_params):

        #self.get_logger().debug("unpausing...")
        #self.unpause()

        #self.dwb_client.get_dwb_params()
        self.dwb_client.send_params_action(dwb_params)
        time.sleep(0.14)

        #self.dwb_client.get_dwb_params()

        #self.get_logger().debug("pausing...")
        #self.pause()

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

        goal_info, robot_pose = self.process_odom(sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]
        self.min_obstacle_distance = min_obstacle_distance

        return lidar_measurements, goal_info, robot_pose, collision

    def process_odom(self, odom):

        goal_distance = math.sqrt(
            (self.goal_pose[0]-odom[0])**2
            + (self.goal_pose[1]-odom[1])**2)

        path_theta = math.atan2(
            self.goal_pose[1]-odom[1],
            self.goal_pose[0]-odom[0])

        goal_angle = path_theta - odom[2]

        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        goal_info = [goal_distance, goal_angle]
        robot_pose = [odom[0], odom[1], odom[2]]

        return goal_info, robot_pose

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        if self.navigator.isNavComplete():
            result = self.check_navigation()
            if (result == NavigationResult.FAILED or result == NavigationResult.CANCELED) and not self.n_navigation_aborted == -1:
                self.send_goal(self.goal_pose)
                self.n_navigation_aborted = self.n_navigation_aborted +1
                if self.n_navigation_aborted == 20:
                    self.navigator.cancelNav()
                    subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 1}'",
                    shell=True,
                    stdout=subprocess.DEVNULL
                    )
                    self.n_navigation_aborted = -1
                    self.get_logger().info('Navigation aborted more than 10 times navigation in pause till next episode')  
                    return True, "nav2 failed"  
                
            if result == NavigationResult.SUCCEEDED:
                self.get_logger().info('Goal')
                return True, "goal"

        if  collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info('Collision')
                return True, "collision"
            else:
                return False, "collision"

        #if result == NavigationResult.SUCCEEDED:
        # if goal_info[0] < 0.40:
        #     self.get_logger().info('Goal')
        #     time.sleep(1.5)
        #     result = self.check_navigation()
        #     return True, "goal"

        if self.episode_step >= self.timeout_steps:
            self.get_logger().info('Timeout')
            print('step : ', self.episode_step)
            return True, "timeout"

        return False, "None"

    def check_navigation(self,):
        result = self.navigator.getResult()
        if result == NavigationResult.SUCCEEDED:
            print('Goal succeeded!')
        elif result == NavigationResult.CANCELED:
            print('Goal was canceled!')
        elif result == NavigationResult.FAILED:
            print('Goal failed!')
        elif result == NavigationResult.UNKNOWN:
            print('Navigation Result UNKNOWN!')
        return result

    def get_reward(self,lidar_measurements, goal_info, robot_pose, done, event):
        """
        """
        #dist_reward = (self.previous_goal_info[0] - goal_info[0])*30 
        #yaw_reward = (1-2*math.sqrt(math.fabs(goal_info[1]/math.pi)))*0.4

        p_t = np.array([robot_pose[0], robot_pose[1]], dtype=np.float32)
        p_tp1 = np.array([self.previous_robot_pose[0], self.previous_robot_pose[1]], dtype=np.float32)
        goal_pose = np.asarray(self.goal_pose, dtype=np.float32)

        cp = 3
        cc = 2
        Rp = np.dot((p_tp1 - p_t), (goal_pose - p_t)) / goal_info[0]
        Rc = -1/self.min_obstacle_distance
        reward = cp*Rp + cc*Rc

        self.get_logger().debug('Rp: ' +str(Rp))
        self.get_logger().debug('Rc: ' +str(Rc))
        self.get_logger().debug('sparse reward: '+str(reward))

        if event == "goal":
            reward += 5
        elif event == "collision":
            reward += -5
        elif event == "None":
            reward += -1

        self.get_logger().debug('total reward: ' +str(reward))

        return reward

    def get_observation(self, lidar_measurements, goal_info, robot_pose, dwb_params):
        # goal angle
        state_list = []
        
        # lidar points
        for point in lidar_measurements:
            state_list.append(float(point))

        state_list.append(goal_info[1])

        # DWB previous parameters
        self.get_logger().debug('dwb_params : '+str(dwb_params))
        state_list.extend(dwb_params)
        state = np.array(state_list, dtype = np.float32)
        #print('State: ', state)
        self.get_logger().debug('state shape: '+str(state.shape))

        return state

    def update_state(self,lidar_measurements, goal_info, robot_pose, dwb_params, done, event):
        """
        """
        self.episode_step += 1
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose
        self.previous_dwb_params = dwb_params

        if done:
            self.init_step = True
            self.episode_step = 0

    def reset(self, n_episode):
        """
        """
        self.episode = n_episode
        self.get_logger().info("Initializing new episode ...\n")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")

        dwb_params = self.init_dwb_params
        _,_,_, = self._step(dwb_params,reset_step = True)
        observation,_,_, = self._step(dwb_params)

        return observation
    
    def new_episode(self):
        """
        """
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()

        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_world_client.call_async(req)
        
        if self.episode % self.change_episode == 0.:
            self.index = int(np.random.uniform()*len(self.poses)) -1 

        self.get_logger().debug("Respawing robot ...")
        self.respawn_robot(self.index)
    
        self.get_logger().debug("Respawing goal ...")
        self.respawn_goal(self.index)

        self.get_logger().debug("Resetting navigator ...")
        self.reset_navigator(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):

        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y, yaw = tuple(self.poses[index])

        self.get_logger().info("New robot pose: (x,y,yaw) : " + str(x)+' '+str(y))

        pose = "'{state: {name: '"+self.robot_name+"',pose: {position: {x: "+str(x)+",y: "+str(y)+",z: 0.1}}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.2)

    def respawn_goal(self, index):
        # GET GOAL
        if self.episode <= self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        # RESPAWN GOAL in GAZEBO
        position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])
        pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.2)

    def reset_navigator(self, index):
        init_pose = PoseStamped()
        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y, yaw = tuple(self.poses[index])

        z = math.sin(yaw/2)
        w = math.cos(yaw/2)

        init_pose.header.frame_id = 'odom'
        init_pose.pose.position.x = x
        init_pose.pose.position.y = y
        init_pose.pose.position.z = 0.0
        init_pose.pose.orientation.x = 0.0
        init_pose.pose.orientation.y = 0.0
        init_pose.pose.orientation.z = z
        init_pose.pose.orientation.w = w
        if self.n_navigation_aborted == -1:
            subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 0}'",
                    shell=True,
                    stdout=subprocess.DEVNULL
                    )
        self.n_navigation_aborted = 0
        #self.navigator.setInitialPose(init_pose)
        self.navigator.clearAllCostmaps()
        self.navigator.waitUntilNav2Active()
        
        self.send_goal(self.goal_pose)


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

        self.navigator.goToPose(goal_pose)
        #self.goal_pub.publish(goal_pose)

    def get_random_goal(self):

        x = random.randrange(-29, 38) / 10.0
        y = random.randrange(-38, 38) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]
            
        self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
        self.goal_pose = [x, y]

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