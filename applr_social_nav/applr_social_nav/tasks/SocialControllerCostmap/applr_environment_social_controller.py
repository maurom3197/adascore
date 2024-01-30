#!/usr/bin/env python3

import os
import numpy as np
from numpy import savetxt
import math
import json
import random
import sys
import time
import datetime
import yaml
import logging
from pathlib import Path

from geometry_msgs.msg import Pose,PoseStamped
from gazebo_msgs.srv import SetEntityState
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.utils.env_utils import *

from applr_social_nav.utils.nav_utils import *
from applr_social_nav.utils.sfm import SocialForceModel

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterValue
from pic4rl.sensors import Sensors
from pic4rl.testing.nav_metrics import Navigation_Metrics

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from people_msgs.msg import People


class Pic4rlEnvironmentAPPLR(Node):
    def __init__(self):
        super().__init__('pic4rl_env_applr')
        self.declare_parameter("package_name", "applr_social_nav")
        self.declare_parameter("main_params_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        goals_path = os.path.join(
            get_package_share_directory(self.package_name), "goals_and_poses"
        )
        self.main_params_path = self.get_parameter(
            "main_params_path").get_parameter_value().string_value
        training_params_path = self.get_parameter(
            "training_params_path").get_parameter_value().string_value
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), 
            'models/goal_box/model.sdf'
            )
        
        with open(training_params_path, 'r') as train_param_file:
            training_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(
            namespace   = '',
            parameters  = [
                ("mode", rclpy.Parameter.Type.STRING),
                ("data_path", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ('agents_config', rclpy.Parameter.Type.STRING),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
                ('laser_param.total_points', rclpy.Parameter.Type.INTEGER),
                ("sensor", rclpy.Parameter.Type.STRING),
                ("use_localization", rclpy.Parameter.Type.BOOL),
                ]
            )
        
        self.mode           = self.get_parameter(
            'mode').get_parameter_value().string_value
        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path      = os.path.join(goals_path, self.data_path)
        print(training_params["--change_goal_and_pose"])
        self.change_episode = int(training_params["--change_goal_and_pose"])
        self.starting_episodes = int(training_params["--starting_episodes"])
        self.timeout_steps = int(training_params["--episode-max-steps"])
        self.robot_name     = self.get_parameter(
            'robot_name').get_parameter_value().string_value
        self.agents_config     = self.get_parameter(
            'agents_config').get_parameter_value().string_value
        self.goal_tolerance     = self.get_parameter(
            'goal_tolerance').get_parameter_value().double_value
        self.params_update_freq   = self.get_parameter(
            'update_frequency').get_parameter_value().double_value
        self.lidar_distance = self.get_parameter(
            'laser_param.max_distance').get_parameter_value().double_value
        self.lidar_points   = self.get_parameter(
            'laser_param.num_points').get_parameter_value().integer_value
        self.total_points   = self.get_parameter(
            'laser_param.total_points').get_parameter_value().integer_value
        self.sensor_type = self.get_parameter(
            "sensor").get_parameter_value().string_value
        self.use_localization = self.get_parameter(
            "use_localization").get_parameter_value().bool_value
        
        # create Sensor class to get and process sensor data
        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)

        log_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', training_params["--logdir"])

        # create log dir 
        self.logdir = create_logdir(
            training_params["--policy"], self.sensor_type, log_path
        )
        self.get_logger().info(f"Logdir: {self.logdir}")

        if "--model-dir" in training_params:
            self.model_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', training_params["--model-dir"])

        if "--rb-path-load" in training_params:
            self.rb_path_load = os.path.join(get_package_share_directory(self.package_name),'../../../../', training_params["--rb-path-load"])

        self.create_clients()
        self.is_paused = None
        self.unpause()

        self.spin_sensors_callbacks()

        # init goal publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)

        self.sfm = SocialForceModel(self, self.agents_config)

        self.episode_step = 0
        self.previous_twist = None
        self.previous_event = "None"
        self.prev_nav_state = "unknown"
        self.simulation_restarted = 0
        self.failure_counter = 0
        self.episode = 0
        self.collision_count = 0
        self.min_obstacle_distance = 12.0
        self.t0 = 0.0
        self.evaluate = False
        self.index = -1
        self.people_state = []
        self.k_people = 4
        self.min_people_distance = 10.0
        self.max_person_dist_allowed = 5.0

        self.initial_pose, self.goals, self.poses, self.agents = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]
        
        self.init_nav_params = [0.6,    # max vel_x
                                1.5,    # max vel theta
                                1.7,    # sim_time
                                0.02,   # scale BaseObstacle
                                32.0,   # scale PathDist
                                24.0,   # scale GoalDist
                                32.0,    # scale PathAlign
                                24.0,    # scale GoalAlign
                                0.55,   # inflation_radius
                                0.25,   # covariance height
                                0.25,   # covariance width
                                0.25,   # covariance still
                                ]
        
        self.n_navigation_end = 0
        self.navigator = BasicNavigator()

        self.get_logger().info(f'Gym mode: {self.mode}')
        if self.mode == "testing":
            self.nav_metrics = Navigation_Metrics(self.logdir)

        self.get_logger().info("PIC4RL_Environment: Starting process")
        self.get_logger().info("Navigation params update at: " + str(self.params_update_freq)+' Hz')

    def step(self, action, episode_step=0):
        """
        """
        self.get_logger().debug("Env step : " + str(episode_step))
        self.episode_step = episode_step

        self.get_logger().debug("Action received (nav2 params): "+str(action))
        params = action.tolist()

        observation, reward, done = self._step(params)
        info = None

        return observation, reward, done, info

    def _step(self, nav_params=None, reset_step = False):
        """
        """
        self.get_logger().debug("sending action...")
        self.send_action(nav_params)

        self.spin_sensors_callbacks()
        self.get_logger().debug("getting sensor data...")
        lidar_measurements, goal_info, robot_pose, robot_velocity, collision = self.get_sensor_data()
        self.get_logger().debug("getting people data...")
        people_state, people_info = self.get_people_state(robot_pose, robot_velocity)
        wr, wp = self.sfm.computeSocialWork()
        social_work = wr + wp

        if not reset_step:
            self.get_logger().debug("checking events...")
            done, event = self.check_events(lidar_measurements, goal_info, robot_pose, collision)

            self.get_logger().debug("getting reward...")
            reward = self.get_reward(lidar_measurements, goal_info, robot_pose, people_state, social_work, done, event)

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(lidar_measurements, goal_info, robot_pose, people_state, nav_params)
        else:
            reward = None
            observation = None
            done = False
            event = 'None'

        self.update_state(lidar_measurements, goal_info, robot_pose, people_state, nav_params, done, event)

        if done:
            self.navigator.cancelTask()
            # lifecyclePause(navigator=self.navigator)
            time.sleep(1.0)

        return observation, reward, done

    def spin_sensors_callbacks(self):
        """
        """
        self.get_logger().debug("spinning for sensor_msg...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            self.get_logger().debug("None in sensor_msg... spinning again...")
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def get_goals_and_poses(self):
        """ """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"], data["agents"]

    def get_sensor_data(self):
        """
        """
        sensor_data = {}
        sensor_data["scan"], min_obstacle_distance, collision = self.sensors.get_laser(min_obstacle_distance=True)
        sensor_data["odom"], sensor_data["velocity"] = self.sensors.get_odom(vel=True)

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
        velocity = sensor_data["velocity"]

        return lidar_measurements, goal_info, robot_pose, velocity, collision

    def send_action(self, params):
        """
        """
        costmap_params = params[-4:]
        controller_params = params[:-4]
        # print(costmap_params)
        # print(controller_params)

        self.set_costmap_params(costmap_params)
        self.set_controller_params(controller_params)

        # Regulate the step frequency of the environment
        action_hz, t1 = compute_frequency(self.t0)
        self.t0 = t1
        if action_hz > self.params_update_freq:
            frequency_control(self.params_update_freq)
            self.get_logger().debug('Sending action at '+str(action_hz))

        # If desired to get params
        #self.get_costmap_params()
        #self.get_controller_params()

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """
        """
        # get action feedback from navigator
        feedback = self.navigator.getFeedback()
        #self.get_logger().debug('Navigator feedback: '+str(feedback))
        # check if navigation is complete
        if self.navigator.isTaskComplete():
            result = check_navigation(self.navigator)
            if (result == TaskResult.FAILED or result == TaskResult.CANCELED):
                self.send_goal(self.goal_pose)
                time.sleep(1.0)
                self.n_navigation_end = self.n_navigation_end +1
                if self.n_navigation_end == 50:
                    self.get_logger().info('Navigation aborted more than 50 times... pausing Nav till next episode.') 
                    self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: 'Nav failed'")
                    logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Nav failed") 
                    self.prev_nav_state = "nav2_failed"
                    return True, "nav2_failed"  
            if result == TaskResult.SUCCEEDED:
                if self.prev_nav_state == "goal":
                    self.get_logger().info('uncorrect goal status detected... resending goal.') 
                #     self.send_goal(self.goal_pose)
                #     return False, "None"

                self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
                logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
                self.prev_nav_state = "goal"
                return True, "goal"
            self.prev_nav_state = "unknown"

        else:
            self.prev_nav_state = "unknown"

        # check collision
        if  collision or self.min_people_distance < 0.50:
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

    def get_reward(self,lidar_measurements, goal_info, robot_pose, people_state, social_work, done, event):
        """
        """
        # Distance Reward
        #dist_reward = (self.previous_goal_info[0] - goal_info[0])*30 

        # p_t = np.array([robot_pose[0], robot_pose[1]], dtype=np.float32)
        # p_tp1 = np.array([self.previous_robot_pose[0], self.previous_robot_pose[1]], dtype=np.float32)
        # goal_pose = np.asarray(self.goal_pose, dtype=np.float32)

        # Heading Reward
        ch = 0.3
        #Rh = np.dot((p_tp1 - p_t), (goal_pose - p_t)) / goal_info[0] #heading reward v.1
        Rh = (1-2*math.sqrt(math.fabs(goal_info[1]/math.pi))) #heading reward v.2
        
        # Social Disturbance Reward
        #avg_people_dist = np.mean(people_state[:,0])
        #Rs = -1/avg_people_dist # people distance reward
        Rp = 0.
        if self.min_people_distance < 1.5: #1.2
            Rp += -1/self.min_people_distance # personal space reward
        cp = 2.0 #1.5

        # Social work
        Rs = -social_work
        cs = 15. #15.0

        # Total Reward
        reward = ch*Rh + cs*Rs + cp*Rp
        #reward = cs*Rs

        self.get_logger().debug('Goal Heading Reward Rh: ' +str(ch*Rh))
        self.get_logger().debug('People proxemics reward Rp: ' +str(cp*Rp))
        self.get_logger().debug('Social work reward Rs: ' +str(cs*Rs))
        self.get_logger().debug('sparse reward: '+str(reward))

        if event == "goal":
            reward += 0 
        elif event == "collision":
            reward += -200
        elif event == "None":
            reward += -1.0

        self.get_logger().debug('total reward: ' +str(reward))

        return reward

    def get_observation(self, lidar_measurements, goal_info, robot_pose, people_state, nav_params):
        
        state_list = []

        # goal info
        state_list.append(goal_info[1])
        state_list.append(goal_info[0])

        # costmap previous parameters
        state_list.extend(nav_params)

        # People info
        people_state = people_state.flatten().tolist()
        state_list.extend(people_state)

        # lidar points
        for point in lidar_measurements:
            state_list.append(float(point))

        state = np.array(state_list, dtype = np.float32)
        #self.get_logger().debug('goal angle: '+str(goal_info[1]))
        #self.get_logger().debug('min obstacle lidar_distance: '+str(self.min_obstacle_distance))
        #self.get_logger().debug('costmap_params : '+str(costmap_params))
        #self.get_logger().debug('state shape: '+str(state.shape))
        #self.get_logger().debug('state=[goal,params,people,lidar]: '+str(state))

        return state

    def update_state(self,lidar_measurements, goal_info, robot_pose, people_state, nav_params, done, event):
        """
        """
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose
        self.people_state = people_state
        self.previous_nav_params = nav_params
        self.previous_event = event

    # def restart_simulation(self,):
    #     """
    #     """
    #     self.get_logger().debug("pausing gazebo...")
    #     self.pause()
    #     # restart_gazebo(self.gazebo_client)
    #     restart_nav2()
    #     self.get_logger().debug("Creating parameters clients...")
    #     self.create_clients()

    #     self.get_logger().debug("unpausing gazebo...")
    #     self.unpause()
    #     time.sleep(5.0)
        
    #     self.navigator = BasicNavigator()
    #     time.sleep(5.0)
    #     self.sensors = Sensors(self)
    #     self.spin_sensors_callbacks()
    #     self.index = 0
    #     self.n_navigation_end = 0

    def reset(self, n_episode, tot_steps, evaluate=False):
        """
        """
        if self.mode == "testing":
            self.nav_metrics.calc_metrics(n_episode, self.initial_pose, self.goal_pose)
            self.nav_metrics.log_metrics_results(n_episode)
            self.nav_metrics.save_metrics_results(n_episode)

        self.episode = n_episode
        self.evaluate = evaluate

        self.get_logger().debug("pausing...")
        self.pause()

        logging.info(f"Total_episodes: {'evaluate' if evaluate else n_episode}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n")
        
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")
        self.new_episode()
        #self.simulation_restarted = 0

        self.get_logger().debug("unpausing...")
        self.unpause()

        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        nav_params = self.init_nav_params
        _,_,_, = self._step(nav_params,reset_step = True)
        observation,_,_, = self._step(nav_params)
        return observation
    
    def new_episode(self):
        """
        """
        # self.get_logger().debug("Resetting simulation ...")
        # req = Empty.Request()

        # while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.reset_world_client.call_async(req)
        
        if self.episode % self.change_episode == 0.:
            #self.index = int(np.random.uniform()*len(self.poses)) -1 
            self.index += 1
            if self.index == len(self.goals):
                self.index = 0
        
        self.get_logger().debug("Respawning robot ...")
        self.respawn_robot(self.index)

        if self.episode % 30 == 0.:
            self.get_logger().debug("Respawning all agents ...")
            self.respawn_agents(all=True)
    
        self.get_logger().debug("Respawning goal ...")
        self.respawn_goal(self.index)

        if self.is_paused:
            self.get_logger().debug("unpausing...")
            self.get_logger().debug("This is done to avoid a bug in the navigator")
            self.unpause()
            time.sleep(1.0)
            self.get_logger().debug("pausing...")
            self.pause()

        self.get_logger().debug("Resetting navigator ...")
        self.reset_navigator(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):
        """
        """
        x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        self.set_entity_state(self.robot_name, [x, y, 0.07], [qz, qw])

    def respawn_agents(self, all=False):
        """
        """
        if self.index in [0,1]:
            agents2reset = [1]
        elif self.index in [2,3,4]:
            agents2reset = [2,3]
        elif self.index > 4:
            agents2reset = [3,4,5,9]
        elif all:
            agents2reset = list(range(1,len(self.agents)+1))
        else:
            return
            
        for agent in agents2reset:
            x, y , yaw = tuple(self.agents[agent-1])

            self.get_logger().info(f"Respawning Agent at pose [x,y,yaw]: {[x, y, yaw]}")
            agent_name = "agent"+str(agent)
            self.set_entity_state(agent_name, [x, y, 1.50])

    def respawn_goal(self, index):
        """
        """
        if self.episode < self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        # self.set_entity_state('goal', [self.goal_pose[0], self.goal_pose[1], 0.01])

    def reset_navigator(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y, yaw = tuple(self.poses[index])

        z = math.sin(yaw/2)
        w = math.cos(yaw/2)

        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = x
        initial_pose.pose.position.y = y
        initial_pose.pose.orientation.z = z
        initial_pose.pose.orientation.w = w
        
        if self.use_localization:
            self.navigator.setInitialPose(initial_pose)

        if not self.simulation_restarted == 1:
            self.get_logger().debug("Restarting LifeCycleNodes...")
            # lifecycleResume(navigator=self.navigator)

            self.n_navigation_end = 0

        self.get_logger().debug("wait until Nav2Active...")
        if self.use_localization:
            localizer = 'amcl'
        else:
            localizer = 'controller_server'
        self.navigator.waitUntilNav2Active(navigator='bt_navigator', localizer=localizer)

        # self.waitUnitlInitialPoseIsSet(initial_pose)
        self.get_logger().debug("Clearing all costmaps...")
        self.navigator.clearAllCostmaps()
        time.sleep(1.0)
 
        self.get_logger().debug("Sending goal ...")
        self.send_goal(self.goal_pose)
        time.sleep(1.0)

    def get_people_state(self, robot_pose, robot_velocity):
        """
        """
        # Spin once to get the people message
        rclpy.spin_once(self)

        people_state, people_info, min_people_distance = self.sfm.get_people(robot_pose, robot_velocity)
        self.min_people_distance = min_people_distance
        self.get_logger().debug('Min people distance: '+str(min_people_distance))

        return people_state, people_info

    ### COSTMAP PARAMS CLIENTS METHODS ###
    def send_set_request_global(self, param_values):
        self.set_req_global.parameters = [
                        Parameter(name='social_layer.covariance_front_height', value=param_values[1]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_front_width', value=param_values[2]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_when_still', value=param_values[3]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_right_width', value=param_values[1]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_right_width', value=param_values[2]).to_parameter_msg(),
                        Parameter(name='inflation_layer.inflation_radius', value=param_values[0]).to_parameter_msg()
                                              ]
        future = self.set_cli_global.call_async(self.set_req_global)
        return future

    def send_set_request_local(self, param_values):
        self.set_req_local.parameters = [
                        Parameter(name='social_layer.covariance_front_height', value=param_values[1]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_front_width', value=param_values[2]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_when_still', value=param_values[3]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_right_height', value=param_values[1]).to_parameter_msg(),
                        Parameter(name='social_layer.covariance_right_width', value=param_values[2]).to_parameter_msg(),
                        Parameter(name='inflation_layer.inflation_radius', value=param_values[0]).to_parameter_msg()
                                              ]
        future = self.set_cli_local.call_async(self.set_req_local)
        return future

    def send_set_request_controller(self, param_values):
        self.set_req_controller.parameters = [Parameter(name='FollowPath.max_vel_x', value=param_values[0]).to_parameter_msg(),
                                              Parameter(name='FollowPath.max_vel_theta', value=param_values[1]).to_parameter_msg(),
                                              #Parameter(name='FollowPath.vx_samples', value=param_values[2]).to_parameter_msg(),
                                              #Parameter(name='FollowPath.vtheta_samples', value=param_values[3]).to_parameter_msg()
                                              Parameter(name='FollowPath.sim_time', value=param_values[2]).to_parameter_msg(),
                                              Parameter(name='FollowPath.BaseObstacle.scale', value=param_values[3]).to_parameter_msg(),
                                              Parameter(name='FollowPath.PathDist.scale', value=param_values[4]).to_parameter_msg(),
                                              Parameter(name='FollowPath.GoalDist.scale', value=param_values[5]).to_parameter_msg()
                                              ]
        future = self.set_cli_controller.call_async(self.set_req_controller)
        return future

    def set_costmap_params(self, costmap_params):
        self.get_logger().debug('setting costmap params to: '+str(costmap_params))

        self.set_req_local = SetParameters.Request()

        future = self.send_set_request_local(costmap_params)
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().debug(
                'Result %s' %
                (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug(
                'Service call failed %r' % (e,))

    def set_controller_params(self, controller_params):
        self.get_logger().debug('setting controller params to: '+str(controller_params))
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

    def get_goal(self, index):
        # get goal from predefined list
        self.goal_pose = self.goals[index]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose))

    def send_goal(self,pose):
        # Set the robot's goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = pose[0]
        goal_pose.pose.position.y = pose[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        # self.goal_pub.publish(goal_pose)
        self.navigator.goToPose(goal_pose)
        
    def get_random_goal(self):

        x = random.randrange(-35, 30) / 10.0
        y = random.randrange(-88, 28) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]
            
        self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
        self.goal_pose = [x, y]

    def compute_frequency(self,):
        t1 = time.perf_counter()
        step_time = t1-self.t0
        self.t0 = t1
        action_hz = 1./(step_time)
        self.get_logger().debug('Sending action at '+str(action_hz))

    def pause(self):
        self.is_paused = True

        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pause_physics_client.call_async(req) 

    def unpause(self):
        self.is_paused = False

        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.unpause_physics_client.call_async(req)

    def set_entity_state(self, entity_name, position, orientation = [0.0, 1.0]):
        """
        """
        req = SetEntityState.Request()

        req.state.name = f'{entity_name}'
        req.state.pose.position.x = position[0]
        req.state.pose.position.y = position[1]
        req.state.pose.position.z = position[2]
        req.state.pose.orientation.z = orientation[0]
        req.state.pose.orientation.w = orientation[1]
        
        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        success = False
        while not success:
            future = self.set_entity_state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                success = future.result().success
                self.get_logger().debug(f'Result of setting entity state: {success}')
            time.sleep(2.0)
        return


    # Global Costmap get_parameter #
    def send_get_request_global(self):
        self.get_req_global.names = [
            'social_layer.covariance_front_height',
            'social_layer.covariance_front_width',
                                    ]
        future = self.get_cli_global.call_async(self.get_req_global)
        return future

    # Local Costmap get_parameter #
    def send_get_request_local(self):
        self.get_req_local.names = [
            'social_layer.covariance_front_height',
            'social_layer.covariance_front_width'
        ]
        future = self.get_cli_local.call_async(self.get_req_local)
        return future

    def send_get_request_controller(self):
        self.get_req_controller.names = [
            'FollowPath.max_vel_x',
            'FollowPath.max_vel_theta'
            #'FollowPath.vx_samples',
            #'FollowPath.vtheta_samples'
        ]
        future = self.get_cli_controller.call_async(self.get_req_controller)
        return future

    def get_costmap_params(self,):

        future = self.send_get_request_global()
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().info(
                    'Result %s %s' %(
                    get_response.values[0].double_value,
                    get_response.values[1].double_value
                    ))

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        future = self.send_get_request_local()

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    get_response = future.result()
                    self.get_logger().info(
                        'Result %s %s' %(
                        get_response.values[0].double_value,
                        get_response.values[1].double_value
                        ))
                except Exception as e:
                    self.get_logger().info(
                        'Service call failed %r' % (e,))
                break

    def get_controller_params(self,):
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
                    #get_response.values[6].double_value
                    ))

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

    def create_clients(self,):

        # self.set_cli_global = self.create_client(SetParameters, '/global_costmap/global_costmap/set_parameters')
        # while not self.set_cli_global.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.set_req_global = SetParameters.Request()

        self.set_cli_local = self.create_client(SetParameters, '/local_costmap/local_costmap/set_parameters')
        while not self.set_cli_local.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_local = SetParameters.Request()

        # # create Controller parameter client
        self.get_cli_controller = self.create_client(GetParameters, '/controller_server/get_parameters')
        while not self.get_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_req_controller = GetParameters.Request()

        self.set_cli_controller = self.create_client(SetParameters, '/controller_server/set_parameters')
        while not self.set_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_controller = SetParameters.Request()

        # create reset world client 
        self.reset_world_client = self.create_client(Empty, 'reset_simulation')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')
        self.set_entity_state_client = self.create_client(SetEntityState, 'test/set_entity_state')