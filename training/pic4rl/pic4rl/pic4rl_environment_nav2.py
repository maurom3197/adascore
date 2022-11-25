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
import datetime
import yaml
import logging
from pathlib import Path

from geometry_msgs.msg import Pose,PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.generic_sensor import Sensors

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterValue

from nav2_simple_commander.robot_navigator import BasicNavigator, NavigationResult

class Pic4rlEnvironmentAPPLR(Node):
    def __init__(self):
        super().__init__('pic4rl_env_applr')
        #rclpy.logging.set_logger_level('pic4rl_env_applr', 10)

        goals_path      = os.path.join(
            get_package_share_directory('pic4rl'), 'goals_and_poses')
        main_params_path  = os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'main_params.yaml')
        training_params_path= os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'training_params.yaml')
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), 
            'models/goal_box/model.sdf'
            )
        
        with open(main_params_path, 'r') as main_params_file:
            main_params = yaml.safe_load(main_params_file)['main_node']['ros__parameters']
        with open(training_params_path, 'r') as train_param_file:
            training_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(
            namespace   = '',
            parameters  = [
                ('data_path', main_params['data_path']),
                ('change_goal_and_pose', training_params['--change_goal_and_pose']),
                ('starting_episodes', training_params['--starting_episodes']),
                ('timeout_steps', training_params['--episode-max-steps']),
                ('robot_name', main_params['robot_name']),
                ('goal_tolerance', main_params['goal_tolerance']),
                ('update_frequency', configParams['applr_param']['update_frequency']),
                ('lidar_dist', main_params['laser_param']['max_distance']),
                ('lidar_points', main_params['laser_param']['num_points'])
                ]
            )

        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path      = os.path.join(goals_path, self.data_path)
        self.change_episode = self.get_parameter(
            'change_goal_and_pose').get_parameter_value().integer_value
        self.starting_episodes = self.get_parameter(
            'starting_episodes').get_parameter_value().integer_value
        self.timeout_steps  = self.get_parameter(
            'timeout_steps').get_parameter_value().integer_value
        self.robot_name     = self.get_parameter(
            'robot_name').get_parameter_value().string_value
        self.goal_tolerance     = self.get_parameter(
            'goal_tolerance').get_parameter_value().double_value
        self.params_update_freq   = self.get_parameter(
            'update_frequency').get_parameter_value().double_value
        self.lidar_distance = self.get_parameter(
            'lidar_dist').get_parameter_value().double_value
        self.lidar_points   = self.get_parameter(
            'lidar_points').get_parameter_value().integer_value

        # create log dir 
        self.create_logdir(train_params['--policy'], main_param['sensor'], train_params['--logdir'])

        # create Sensor class to get and process sensor data
        self.sensors = Sensors(self)
        self.spin_sensors_callbacks()
        qos = QoSProfile(depth=10)
        # init goal publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)

        # create DWB and Costmap parameter client
        self.get_cli_controller = self.create_client(GetParameters, '/controller_server/get_parameters')
        while not self.get_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_req_controller = GetParameters.Request()

        self.get_cli_costmap = self.create_client(GetParameters, '/local_costmap/local_costmap/get_parameters')
        while not self.get_cli_costmap.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_req_costmap = GetParameters.Request()

        self.set_cli_controller = self.create_client(SetParameters, '/controller_server/set_parameters')
        while not self.set_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_controller = SetParameters.Request()

        self.set_cli_costmap = self.create_client(SetParameters, '/local_costmap/local_costmap/set_parameters')
        while not self.set_cli_costmap.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_costmap = SetParameters.Request()


        # create reset world client 
        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        self.entity_path = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')

        self.episode_step = 0
        self.previous_twist = None
        self.episode = 0
        self.collision_count = 0
        self.min_obstacle_distance = 4.0
        self.t0 = 0.0
        self.evaluate = False
        self.index = 0

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.init_dwb_params = [0.4, 1.5, 20, 20, 0.02, 32.0, 24.0, 0.55]
        self.n_navigation_end = 0
        self.navigator = BasicNavigator()

        self.get_logger().info("PIC4RL_Environment: Starting process")
        self.get_logger().info("Navigation params update at: " + str(self.params_update_freq)+' Hz')

    def step(self, action, episode_step=0):
        """
        """
        self.get_logger().debug("Env step : " + str(episode_step))
        self.episode_step = episode_step

        #self.get_logger().debug("Action received: "+str(action))
        dwb_params = action.tolist()
        #self.get_logger().debug("dwb_params: "+str(dwb_params))
        dwb_params[2] = int(dwb_params[2])
        dwb_params[3] = int(dwb_params[3])

        observation, reward, done = self._step(dwb_params)
        info = None

        return observation, reward, done, info

    def _step(self, dwb_params=None, reset_step = False):
        """
        """
        self.get_logger().debug("sending action...")

        self.send_action(dwb_params)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
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

        if done:
            self.navigator.cancelNav()
            subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 1}'",
            shell=True,
            stdout=subprocess.DEVNULL
            )

        return observation, reward, done

    def spin_sensors_callbacks(self):
        """
        """
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def send_action(self,dwb_params):

        #self.get_logger().debug("unpausing...")
        #self.unpause()

        #self.get_dwb_params()
        self.send_params_action(dwb_params)
        self.compute_frequency()
        self.frequency_control()

        #self.get_dwb_params()

        #self.get_logger().debug("pausing...")
        #self.pause()

    def compute_frequency(self,):
        t1=time.perf_counter()
        step_time = t1-self.t0
        self.t0 = t1
        action_hz = 1./(step_time)
        self.get_logger().debug('Sending action at '+str(action_hz))

    def frequency_control(self):
        #self.get_logger().debug("Sleeping for: "+str(1/self.params_update_freq) +' s')
        time.sleep(1/self.params_update_freq)


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
        # get action feedback from navigator
        feedback = self.navigator.getFeedback()
        #self.get_logger().debug('Navigator feedback: '+str(feedback))
        # check if navigation is complete
        if self.navigator.isNavComplete():
            result = self.check_navigation()
            if (result == NavigationResult.FAILED or result == NavigationResult.CANCELED):
                self.send_goal(self.goal_pose)
                self.n_navigation_end = self.n_navigation_end +1
                if self.n_navigation_end == 20:
                    self.get_logger().info('Navigation aborted more than 20 times... pausing Nav till next episode.') 
                    self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: 'Nav failed'")
                    logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Nav failed") 
                    return True, "nav2 failed"  
                
            if result == NavigationResult.SUCCEEDED:
                self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
                logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
                return True, "goal"

        # check collision
        if  collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        # check timeout steps
        if self.episode_step == self.timeout_steps:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
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
            reward += 10
        elif event == "collision":
            reward += -100
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
        
        state_list.extend(dwb_params)
        state = np.array(state_list, dtype = np.float32)
        #self.get_logger().debug('goal angle: '+str(goal_info[1]))
        #self.get_logger().debug('min obstacle lidar_distance: '+str(self.min_obstacle_distance))
        #self.get_logger().debug('dwb_params : '+str(dwb_params))
        #self.get_logger().debug('state shape: '+str(state.shape))

        return state

    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path,'r'))

        return data["initial_pose"], data["goals"], data["poses"]

    def update_state(self,lidar_measurements, goal_info, robot_pose, dwb_params, done, event):
        """
        """
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose
        self.previous_dwb_params = dwb_params

    def reset(self, n_episode, tot_steps, evaluate=False):
        """
        """
        self.episode = n_episode
        self.evaluate = evaluate
        logging.info(f"Total_episodes: {'evaluate' if evaluate else n_episode}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n")
        print()
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

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

        self.get_logger().debug("Respawning robot ...")
        self.respawn_robot(self.index)
    
        self.get_logger().debug("Respawning goal ...")
        self.respawn_goal(self.index)

        self.get_logger().debug("Resetting navigator ...")
        self.reset_navigator(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")

        position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(0.07)+"}"
        orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
        pose = position+", "+orientation
        state = "'{state: {name: '"+self.robot_name+"',pose: {"+pose+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
            shell=True,
            stdout=subprocess.DEVNULL
            )

    def respawn_goal(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")

        position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])+",z: "+str(0.01)+"}"
        pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )

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
        #if self.n_navigation_end == -1:
            # self.get_logger().debug("Resetting LifeCycleNodes...")
            # subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 3}'",
            #         shell=True,
            #         stdout=subprocess.DEVNULL
            #         )
        self.get_logger().debug("Restarting LifeCycleNodes...")
        subprocess.run("ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes '{command: 0}'",
                shell=True,
                stdout=subprocess.DEVNULL
                )
        time.sleep(0.1)

        self.n_navigation_end = 0
        self.get_logger().debug("Setting navigator initial Pose...")
        self.navigator.setInitialPose(init_pose)
        if self.episode % 10 == 0:
            self.get_logger().debug("Clearing all costmaps...")
            self.navigator.clearAllCostmaps()
        else:
            self.get_logger().debug("Clearing local costmap...")
            self.navigator.clearLocalCostmap()
        self.get_logger().debug("wait until Nav2Active...")
        self.navigator.waitUntilNav2Active()

        self.get_logger().debug("Sending goal ...")
        self.send_goal(self.goal_pose)

    ### DWB PARAMS CLIENTS METHODS ###
    # set_parameter #
    def send_set_request_controller(self, param_values):
        self.set_req_controller.parameters = [Parameter(name='FollowPath.max_vel_x', value=param_values[0]).to_parameter_msg(),
                                              Parameter(name='FollowPath.max_vel_theta', value=param_values[1]).to_parameter_msg(),
                                              Parameter(name='FollowPath.vx_samples', value=param_values[2]).to_parameter_msg(),
                                              Parameter(name='FollowPath.vtheta_samples', value=param_values[3]).to_parameter_msg(),
                                              Parameter(name='FollowPath.BaseObstacle.scale', value=param_values[4]).to_parameter_msg(),
                                              Parameter(name='FollowPath.PathDist.scale', value=param_values[5]).to_parameter_msg(),
                                              Parameter(name='FollowPath.GoalDist.scale', value=param_values[6]).to_parameter_msg()]
        future = self.set_cli_controller.call_async(self.set_req_controller)
        return future

    def send_set_request_costmap(self, param_value):
        self.set_req_costmap.parameters = [Parameter(name='inflation_layer.inflation_radius', value=param_value).to_parameter_msg()]
        future = self.set_cli_costmap.call_async(self.set_req_costmap)
        return future

    def send_params_action(self, dwb_params):
        controller_params = dwb_params[:-1]
        self.get_logger().debug('setting controller_params to: '+str(controller_params))
        costmap_param = dwb_params[-1]
        self.get_logger().debug('setting inflation_radius to: '+str(costmap_param))

        self.set_req_controller = SetParameters.Request()
        future = self.send_set_request_controller(controller_params)
        rclpy.spin_until_future_complete(self, future)

        try:
            get_response = future.result()
            self.get_logger().debug(
                'Result %s' %
                (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug(
                'Service call failed %r' % (e,))

        # while rclpy.ok():
        #     rclpy.spin_once(self)
        #     if future.done():
        #         try:
        #             get_response = future.result()
        #             self.get_logger().info(
        #                 'Result %s' %
        #                 (get_response.results[0].successful))
        #         except Exception as e:
        #             self.get_logger().info(
        #                 'Service call failed %r' % (e,))
        #             break

        self.set_req_costmap = SetParameters.Request()

        future = self.send_set_request_costmap(costmap_param)
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().debug(
                'Result %s' %
                (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug(
                'Service call failed %r' % (e,))

        # while rclpy.ok():
        #     rclpy.spin_once(self)
        #     if future.done():
        #         try:
        #             get_response = future.result()
        #             self.get_logger().info(
        #                 'Result %s' %
        #                 (get_response.results[0].successful))
        #         except Exception as e:
        #             self.get_logger().info(
        #                 'Service call failed %r' % (e,))
        #         break

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

    # DWB get_parameter #
    def send_get_request_controller(self):
        self.get_req_controller.names = ['FollowPath.max_vel_x','FollowPath.max_vel_theta','FollowPath.vx_samples','FollowPath.vtheta_samples','FollowPath.BaseObstacle.scale','FollowPath.PathDist.scale','FollowPath.GoalDist.scale']
        future = self.get_cli_controller.call_async(self.get_req_controller)
        return future

    def send_get_request_costmap(self):
        self.get_req_costmap.names = ['inflation_layer.inflation_radius']
        future = self.get_cli_costmap.call_async(self.get_req_costmap)
        return future

    def get_dwb_params(self,):

        future = self.send_get_request_controller()
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().info(
                    'Result %s %s %s %s %s %s %s' %(
                    get_response.values[0].double_value,
                    get_response.values[1].double_value, 
                    get_response.values[2].integer_value, 
                    get_response.values[3].integer_value,
                    get_response.values[4].double_value,
                    get_response.values[5].double_value,
                    get_response.values[6].double_value
                    ))

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        future = self.send_get_request_costmap()

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    get_response = future.result()
                    self.get_logger().info(
                        'Result %s ' %
                        (get_response.values[0].double_value))
                except Exception as e:
                    self.get_logger().info(
                        'Service call failed %r' % (e,))
                break
    
    def create_logdir(self, policy, sensor, logdir):
                """
                """
                self.logdir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{sensor}_{policy}/"
                Path(os.path.join(logdir, self.logdir)).mkdir(parents=True, exist_ok=True)
                logging.basicConfig(
                    filename=os.path.join(logdir, self.logdir, 'screen_logger.log'), 
                    level=logging.INFO)