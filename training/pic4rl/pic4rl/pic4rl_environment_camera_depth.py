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
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from pic4rl.generic_sensor import Sensors

class Pic4rlEnvironmentCamera(Node):
    def __init__(self):
        """##########
        Environment initialization
        ##########"""
        super().__init__('pic4rl_env_lidar')
        # rclpy.logging.set_logger_level('pic4rl_env_lidar', 10)
        goals_path = os.path.join(get_package_share_directory('pic4rl'), 'goals_and_poses')
        self.world_path = os.path.join(get_package_share_directory('gazebo_sim'), 'worlds', 'hospital.world')
        configFilepath = os.path.join(get_package_share_directory('pic4rl'), 'config', 'ros_params.yaml')
        with open(configFilepath, 'r') as file:
            configParams = yaml.safe_load(file)['pic4rl_training']['ros__parameters']

        self.declare_parameters(namespace='',
        parameters=[
            ('data_path', configParams['data_path']),
            ('change_goal_and_pose', configParams['change_goal_and_pose']),
            ('timeout_steps', configParams['timeout_steps']),
            ('collision_check', configParams['collision_check']),
            ('visual_data', configParams['visual_data']),
            ('features', configParams['features'])
            ])

        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.data_path = os.path.join(goals_path, self.data_path)
        self.change_episode = self.get_parameter('change_goal_and_pose').get_parameter_value().integer_value
        self.timeout_steps = self.get_parameter('timeout_steps').get_parameter_value().integer_value
        self.collision_check = self.get_parameter('collision_check').get_parameter_value().double_value
        self.visual_data = self.get_parameter('visual_data').get_parameter_value().string_value
        self.features = self.get_parameter('features').get_parameter_value().double_value

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos)

        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        self.entity_path = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')
        self.change_index = -1
        self.init_step = True
        self.episode_step = 0
        self.starting_episodes = 400
        self.previous_twist = None
        self.robot_flag = False
        self.episode = 0
        self.collision_count = 0

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.get_logger().info("Terminate getting goals")

        self.get_logger().info("PIC4RL_Environment: Starting process")


    """#############
    Trainer overridden functions
    #############"""


    def step(self, action):
        """
        """
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])

        observation, reward, done = self._step(twist)
        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step = False):
        """
        """     
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        lidar_measurements, depth_image, goal_info, robot_pose = self.get_sensor_data()

        self.get_logger().debug("checking events...")
        done, event = self.check_events(lidar_measurements, goal_info, robot_pose)

        if not reset_step:
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(twist, lidar_measurements, goal_info, robot_pose, done, event)

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(twist, depth_image, goal_info, robot_pose)
        
        else:
            reward = None
            observation = None

        # Send observation and reward
        self.update_state(twist, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)

        return observation, reward, done

    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path,'r'))

        return data["initial_pose"], data["goals"], data["poses"]

    def send_action(self,twist):
        """
        """
        #self.get_logger().debug("unpausing...")
        self.unpause()

        #self.get_logger().debug("publishing twist...")
        self.cmd_vel_pub.publish(twist)

        time.sleep(0.1)

        #self.get_logger().debug("pausing...")
        self.pause()

    def get_sensor_data(self):
        """
        """
        sensor_data = {"depth":None}
        sensor_data["scan"] = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()
        sensor_data["depth"] = self.sensors.get_depth()

        if sensor_data["scan"] is None:
            sensor_data["scan"] = np.squeeze(np.ones((1,36))*15.0).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0,0.0,0.0]
        if sensor_data["depth"] is None:
             sensor_data["depth"] = np.ones((self.height,self.width,1))*self.cutoff

        goal_info, robot_pose = self.process_odom(sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]
        depth_image = sensor_data["depth"]

        return lidar_measurements, depth_image, goal_info, robot_pose

    def process_odom(self, odom):
        """
        """
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

    def check_events(self, lidar_measurements, goal_info, robot_pose):
        """
        """
        min_range = self.collision_check
        min_lidar = min(lidar_measurements)
        self.get_logger().debug("Min lidar: {}".format(min_lidar))

        ## FOR VINEYARD ONLY ##
        # if math.fabs(robot_pose[2]) > 1.57:
        #     robot_pose[2] = math.fabs(robot_pose[2]) - 3.14
        # yaw_limit = math.fabs(robot_pose[2])-1.4835  #check yaw is less than 85°
        # self.get_logger().debug("Yaw limit: {}".format(yaw_limit))
        # if yaw_limit > 0:
        #     self.get_logger().info('Reverse: yaw too high')
        #     return True, "reverse"

        if  0.05 <  min_lidar < min_range:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info('Collision')
                return True, "collision"
            else:
                return False, "collision"

        if goal_info[0] < 0.45:
            self.get_logger().info('Goal')
            return True, "goal"

        if self.episode_step >= self.timeout_steps:
            self.get_logger().info('Timeout')
            print('step : ', self.episode_step)
            return True, "timeout"

        return False, "None"

    def get_reward(self,twist,lidar_measurements, goal_info, robot_pose, done, event):
        """
        """
        reward = (self.previous_goal_info[0] - goal_info[0])*30 
        yaw_reward = (1-2*math.sqrt(math.fabs(goal_info[1]/math.pi)))*0.4

        reward += yaw_reward

        if event == "goal":
            reward += 1000
        elif event == "collision":
            reward += -200
        self.get_logger().debug(str(reward))

        return reward

    def get_observation(self, twist, depth_image, goal_info, robot_pose):
        """
        """
        if self.visual_data == 'image':
            features = depth_image.flatten()
        # elif self.visual_data == 'feature':
        #     feature = 
        
        state = np.concatenate((goal_info, features))

        return state

    def update_state(self,twist, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
        """
        """
        self.episode_step += 1
        self.previous_twist = twist
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

        if done:
            self.init_step = True
            self.episode_step = 0

    def reset(self, n_episode):

        self.episode = n_episode
        self.get_logger().debug("Environment reset ...")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")

        _,_,_, = self._step(reset_step = True)
        observation,_,_, = self._step()

        return observation
    
    def new_episode(self):
        """
        """
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()

        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_simulation_client.call_async(req)
        
        if int(self.episode/self.change_episode) != self.change_index:
            self.change_index = int(self.episode/self.change_episode)

            self.get_logger().debug("Respawing robot ...")
            index = int(np.random.uniform()*len(self.poses)) -1 
            self.respawn_robot(index)
        
            self.get_logger().debug("Respawing goal ...")
            self.respawn_goal(index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_goal(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        self.get_logger().debug("deleting goal entity...")
        try:
            self.delete_entity('goal')
        except:
            pass

        self.get_logger().debug("respawning goal entity...")
        entity_path=self.entity_path
        entity_name = 'goal'
        initial_pose = Pose()
        initial_pose.position.x = self.goal_pose[0]
        initial_pose.position.y = self.goal_pose[1]

        self.spawn_entity(initial_pose, ntity_name, entity_path)

    def get_goal(self, index):
        """
        """
        self.goal_pose = self.goals[index]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose))

    def get_random_goal(self):
        """
        """
        if self.episode < 6 or self.episode % 25 == 0:
            x = 0.55
            y = 0.55
        else:
            x = random.randrange(-29, 29) / 10.0
            y = random.randrange(-29, 29) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]
            
        self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
        self.goal_pose = [x, y]

    def respawn_robot(self, index):
        """
        """
        if self.robot_flag:
            self.delete_entity("robot")
        self.robot_flag = False

        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y , yaw = tuple(self.poses[index])

        self.get_logger().info("New robot pose: (x,y,yaw) : " + str(self.poses[index]))

        pose = '-x '+str(x)+' -y '+str(y)+' -Y '+str(yaw)
        robot = "robot "
        command = 'ros2 run gazebo_ros spawn_entity.py -entity '+robot+pose+' -topic /robot_description'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        counter = 0
        while process.poll() is None:
            time.sleep(1.0)
            counter += 1
            if counter > 5.0:
                process.terminate()
                self.run_gazebo()
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                counter = 0
        
        self.robot_flag = True

    def run_gazebo(self):
        """
        """
        kill_gazebo = 'pkill -9 gazebo'
        kill_client = 'pkill -9 gzclient'
        kill_server = 'pkill -9 gzserver'
        process_kill_gazebo = subprocess.Popen(kill_gazebo.split(" "), stdout=subprocess.PIPE)
        process_kill_client = subprocess.Popen(kill_client.split(" "), stdout=subprocess.PIPE)
        process_kill_server = subprocess.Popen(kill_server.split(" "), stdout=subprocess.PIPE)
        time.sleep(3.0)
        command_gazebo = 'gazebo --verbose '+self.world_path+' -s libgazebo_ros_init.so -s libgazebo_ros_factory.so'
        self.get_logger().debug("gazebo start: {}".format(command_gazebo))
        process_gazebo = subprocess.Popen(command_gazebo.split(), stdout=subprocess.PIPE)

        time.sleep(30.0)

    def pause(self):
        """
        """
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pause_physics_client.call_async(req) 

    def unpause(self):
        """
        """
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.unpause_physics_client.call_async(req) 
    
    def spawn_entity(self,pose = None, name = None, entity_path = None, entity = None):
        """
        """
        if not pose:
            pose = Pose()
        req = SpawnEntity.Request()
        req.name = name
        if entity_path:
            entity = open(entity_path, 'r').read()
        req.xml = entity
        req.initial_pose = pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)

    def delete_entity(self, entity_name):
        """
        """
        req = DeleteEntity.Request()
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.delete_entity_client.call_async(req)
        self.get_logger().debug('Entity deleting request sent ...')