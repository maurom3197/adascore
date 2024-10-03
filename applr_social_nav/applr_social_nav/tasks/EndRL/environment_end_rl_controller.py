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
from gazebo_msgs.srv import SetEntityState
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from pic4rl.utils.env_utils import *
from pic4rl.testing.nav_metrics import Navigation_Metrics
from people_msgs.msg import People

from applr_social_nav.utils.nav_utils import *
from applr_social_nav.utils.sfm import SocialForceModel

class Pic4rlEnvironmentAPPLR(Node):
    def __init__(self):
        super().__init__("pic4rl_env_applr")
        self.declare_parameter("package_name", "applr_social_nav")
        self.declare_parameter("main_params_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        goals_path = os.path.join(
            get_package_share_directory(self.package_name), "goals_and_poses"
        )
        saved_paths_dir = os.path.join(goals_path, "saved_paths")
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
                ("saved_paths_file", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ('agents_config', rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ('max_lin_vel', rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
                ('laser_param.total_points', rclpy.Parameter.Type.INTEGER),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("sensor", rclpy.Parameter.Type.STRING),
                ("use_local_goal", rclpy.Parameter.Type.BOOL),
            ],
        )

        self.mode           = self.get_parameter(
            'mode').get_parameter_value().string_value
        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path = os.path.join(goals_path, self.data_path)
        print(training_params["--change_goal_and_pose"])
        self.saved_paths_file = self.get_parameter(
            'saved_paths_file').get_parameter_value().string_value
        self.saved_paths_path = os.path.join(saved_paths_dir, self.saved_paths_file)
        self.change_episode = int(training_params["--change_goal_and_pose"])
        self.starting_episodes = int(training_params["--starting_episodes"])
        self.timeout_steps = int(training_params["--episode-max-steps"])
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.agents_config     = self.get_parameter(
            'agents_config').get_parameter_value().string_value
        self.max_lin_vel     = self.get_parameter(
            'max_lin_vel').get_parameter_value().double_value
        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.lidar_distance = (
            self.get_parameter("laser_param.max_distance")
            .get_parameter_value()
            .double_value
        )
        self.lidar_points = (
            self.get_parameter("laser_param.num_points")
            .get_parameter_value()
            .integer_value
        )
        self.total_points   = self.get_parameter(
            'laser_param.total_points').get_parameter_value().integer_value
        self.update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )
        self.use_local_goal =  (
            self.get_parameter("use_local_goal").get_parameter_value().bool_value
        )
        self.bag_process = None
        self.bag_episode = 0

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
        self.hunav_goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)
        
        # init cmd_vel publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "cmd_vel",
            qos)

        self.sfm = SocialForceModel(self, self.agents_config)

        self.episode_step = 0
        self.previous_twist = None
        self.previous_event = "None"
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
        self.previous_local_goal_info = [0.0,0.0]
        self.Lt = 2.5
        self.pind = 0

        self.initial_pose, self.goals, self.poses, self.agents = self.get_goals_and_poses()
        self.get_logger().info(f"Robot initial pose: {str(self.initial_pose)}")
        self.goal_pose = self.goals[0]
        self.local_goal_pose = [0.0, 0.0]

        if self.use_local_goal:
            # Load precomputed global paths from file
            with open(self.saved_paths_path) as json_file:
                self.global_path_dict = json.load(json_file)

        self.get_logger().info(f"Gym mode: {self.mode}")
        # if self.mode == "testing":
        #     self.nav_metrics = Navigation_Metrics(self.logdir)
        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """
        """
        self.get_logger().debug("Env step : " + str(episode_step))
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.episode_step = episode_step
        self.get_logger().debug("Action received: "+str(action))
        observation, reward, done = self._step(twist)
        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step=False):
        """
        """
        self.get_logger().debug("sending action...")
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        (
            lidar_measurements, 
            goal_info, 
            local_goal_info, 
            robot_pose, 
            robot_velocity, 
            collision
        ) = self.get_sensor_data()

        self.get_logger().debug("getting people data...")
        people_state, people_info = self.get_people_state(robot_pose, robot_velocity)
        wr, wp = self.sfm.computeSocialWork()
        social_work = wr + wp

        if not reset_step:
  
            self.get_logger().debug("checking events...")
            done, event = self.check_events(
                goal_info, local_goal_info, robot_pose, collision
            )
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(
                local_goal_info, social_work, robot_velocity, event
            )

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(
                twist, lidar_measurements, local_goal_info, robot_pose, people_state
            )
        else:
            reward = None
            observation = None
            done = False
            event = None

        self.update_state(twist, lidar_measurements, local_goal_info, robot_pose, people_state, done, event)

        return observation, reward, done

    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"], data["agents"]

    def spin_sensors_callbacks(self):
        """
        """
        self.get_logger().debug("spinning for sensor_msg...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            #self.get_logger().debug("None in sensor_msg... spinning again...")
            rclpy.spin_once(self)
        self.get_logger().debug("sensor msgs spinning complete...")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        rclpy.spin_once(self)
        
    def send_action(self, twist):
        """
        """
        
        self.cmd_vel_pub.publish(twist)
        # Regulate frequency of send action if needed
        freq, t1 = compute_frequency(self.t0)
        self.get_logger().debug(f"frequency : {freq}")
        self.t0 = t1
        if freq > self.update_freq:
            frequency_control(self.update_freq)

        # self.get_logger().debug("pausing...")
        # self.pause()

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
        
        if self.local_goal_pose is None:
            self.get_logger().debug("local goal data is None...")
            self.local_goal_pose = [0.0,0.0]

        local_goal_info, _ = process_odom(self.local_goal_pose, sensor_data["odom"])
        
        lidar_measurements = sensor_data["scan"]
        self.min_obstacle_distance = min_obstacle_distance
        velocity = sensor_data["velocity"]

        return lidar_measurements, goal_info, local_goal_info, robot_pose, velocity, collision

    def get_people_state(self, robot_pose, robot_velocity):
        """
        """
        # Spin once to get the people message
        rclpy.spin_once(self)

        people_state, people_info, min_people_distance = self.sfm.get_people(robot_pose, robot_velocity)
        self.min_people_distance = min_people_distance
        self.get_logger().debug('Min people distance: '+str(min_people_distance))

        return people_state, people_info
    
    def check_events(self, goal_info, local_goal_info, robot_pose, collision):
        """
        """
        # check collision
        if collision or self.min_people_distance < 0.50:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        # check goal reached
        if goal_info[0] < self.goal_tolerance and self.goal_pose==self.local_goal_pose:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            return True, "goal"
        
        # check waypoint reached
        if local_goal_info[0] < self.goal_tolerance and self.use_local_goal:
            self.get_logger().debug(f"Local goal reached..")
            self.set_local_goal(robot_pose)
            return False, "local_goal"

        # check timeout steps
        if self.episode_step == self.timeout_steps-1:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
            return True, "timeout"

        return False, "None"

    def get_reward(self, goal_info, social_work, robot_velocity, event):
        """
        """
        # Distance Reward
        # Positive reward if the distance to the goal is decreased
        cd = 1.0
        Rd = (self.previous_local_goal_info[0] - goal_info[0])*20.0
        
        Rd = np.minimum(Rd, 2.0) 
        Rd = np.maximum(Rd, -2.0)

        # Heading Reward
        ch = 0.4
        Rh = (1-2*math.sqrt(math.fabs(goal_info[1]/math.pi))) #heading reward v.2
        
        # Linear Velocity Reward
        cv = 0.25
        Rv = (robot_velocity[0] - self.max_lin_vel)/self.max_lin_vel # velocity reward
        
        # Obstacle Reward
        co = 0.8
        Ro = (self.min_obstacle_distance - self.lidar_distance)/self.lidar_distance # obstacle reward

        # Social Disturbance Reward
        Rp = 0.
        Rp += 1/self.min_people_distance # personal space reward 1.2 m social 3.6
        Rp = -np.minimum(Rp, 1.5)
        cp = 1.25

        # Social work
        Rs = social_work * 10
        Rs = -np.minimum(Rs, 2.0) 
        cs = 1.25

        # Total Reward
        reward = ch*Rh + cs*Rs + cp*Rp + cd*Rd + cv*Rv + co*Ro
        
        self.get_logger().debug('Goal Heading Reward Rh: ' +str(ch*Rh))
        self.get_logger().debug('People proxemics reward Rp: ' +str(cp*Rp))
        self.get_logger().debug('Social work reward Rs: ' +str(cs*Rs))
        self.get_logger().debug('Goal Distance Reward Rd: ' +str(cd*Rd))
        self.get_logger().debug('Velocity Reward Rv: ' +str(cv*Rv))
        self.get_logger().debug('Obstacle Reward Ro: ' +str(co*Ro))
        self.get_logger().debug('Dense reward : ' +str(reward))

        if event == "goal":
            reward += 1000.0
        elif event == "local_goal":
            reward += 50.0
        elif event == "collision":
            reward += -300
        elif event == "timeout":
            reward += -50

        self.get_logger().debug('total reward: ' +str(reward))
        return reward

    def get_observation(self, twist, lidar_measurements, goal_info, robot_pose, people_state):
        """
        """
        state_list = []

        # goal info
        state_list.append(goal_info[1])
        state_list.append(goal_info[0])

        # add previous Twist command
        # previous velocity state
        v = twist.linear.x
        w = twist.angular.z
        state_list.append(v)
        state_list.append(w)

        # People info
        people_state = people_state.flatten().tolist()
        state_list.extend(people_state)

        # lidar points
        for point in lidar_measurements:
            state_list.append(float(point))

        state = np.array(state_list, dtype = np.float32)
        self.get_logger().debug('state=[goal,people,lidar]: '+str(state))
        return state

    def update_state(
        self, twist, lidar_measurements, local_goal_info, robot_pose, people_state, done, event
    ):
        """
        """
        self.previous_twist = twist
        self.previous_lidar_measurements = lidar_measurements
        self.previous_local_goal_info = local_goal_info
        self.previous_robot_pose = robot_pose
        self.people_state = people_state
        self.previous_event = event

    def reset(self, n_episode, tot_steps, evaluate=False):
        """
        """
        # if self.mode == "testing":
        #     self.nav_metrics.calc_metrics(n_episode, self.initial_pose, self.goal_pose)
        #     self.nav_metrics.log_metrics_results(n_episode)
        #     self.nav_metrics.save_metrics_results(n_episode)

        self.episode = n_episode
        self.evaluate = evaluate

        self.get_logger().debug("pausing...")
        self.pause()

        if self.mode == "testing" and self.bag_process is not None:
            self.bag_process.terminate()
            self.bag_process.communicate()
            self.bag_process = None

        logging.info(f"Total_episodes: {'evaluate' if evaluate else n_episode}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n")
        
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")
        self.new_episode()

        if self.mode == "testing":
            Path(os.path.join(self.logdir, 'evaluator/')).mkdir(parents=True, exist_ok=True)
            evaluator_path = str(Path(os.path.join(self.logdir, 'evaluator','metrics')))
            self.get_logger().debug(f"Evaluator metrics file path: {evaluator_path}")
            results = subprocess.run(f"ros2 launch hunav_evaluator hunav_evaluator_launch.py metrics_output_path:={evaluator_path} &",
                shell=True,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                universal_newlines = True 
                )
            self.get_logger().debug(f"Evaluator run output: {results.stdout} {results.stderr}")
            topic_list = " ".join(["/jackal/odom", "/people", '/cost_weights', '/goal_pose', ])
            if self.bag_episode == 0:
                Path(os.path.join(self.logdir, 'bags')).mkdir(parents=True, exist_ok=True)
            self.bag_process = subprocess.Popen(f"ros2 bag record {topic_list} -o {self.logdir}/bags/episode_{self.bag_episode}",
                shell=True,
                stdout=subprocess.DEVNULL
                )
            self.bag_episode += 1

        self.get_logger().debug("unpausing...")
        self.unpause()

        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        _, _, _, = self._step(reset_step=True)
        observation, _, _, = self._step()

        return observation

    def new_episode(self):
        """
        """
        if self.episode % self.change_episode == 0.:
            self.index += 1
            if self.index == len(self.goals):
                self.index = 0
        
        self.get_logger().debug("Respawning robot ...")
        self.respawn_robot(self.index)

        if self.episode % 12 == 0. or self.mode == "testing":
            self.get_logger().debug("Respawning all agents ...")
            if not self.is_paused:
                self.pause()
            if self.mode == "testing":
                self.get_logger().info("Respawning all agents ...")
                self.respawn_agents(all=True)
            else:
                self.respawn_agents(all=True)

        self.get_logger().debug("Respawning goal ...")
        self.respawn_goal(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):
        """
        """
        x, y, yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        self.set_entity_state(self.robot_name, [x, y, 0.07], [qz, qw])

    def respawn_agents(self, all=False):
        """
        """
        if all: # if testing
            agents2reset = list(range(1,len(self.agents)+1))
        # if training in social_nav.world
        elif self.index in range(8):
            agents2reset = [1,2,3,12]
        elif self.index in range(8,18):
            agents2reset = [4,5,6,9,13,14] #2,3 
        elif self.index > 18:
            agents2reset = [7,8,9,10,11]
        else:
            return
            
        for agent in agents2reset:
            x, y , yaw = tuple(self.agents[agent-1])

            self.get_logger().debug(f"Respawning Agent at pose [x,y,yaw]: {[x, y, yaw]}")
            agent_name = "agent"+str(agent)
            self.set_entity_state(agent_name, [x, y, 1.50])

    def respawn_goal(self, index):
        """
        """
        if self.episode < self.starting_episodes:
            self.get_random_goal(index)
        else:
            self.get_goal(index)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")

        if self.use_local_goal:
            self.global_path = self.global_path_dict["Ep"+str(index)+"_poses"]
            x, y, _ = tuple(self.poses[index])
            self.pind = 0
            self.set_local_goal([x,y])
        else:
            self.local_goal_pose = self.goal_pose

    def get_goal(self, index):
        # get goal from predefined list
        self.goal_pose = self.goals[index]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose))
    
    def set_local_goal(self, robot_pose):
        """""
        Compute local goal at a fixed distance on the global path
        """""
        goal_pose = np.array(self.goal_pose)
        local_goal_pose = np.array(self.local_goal_pose)

        if np.linalg.norm(goal_pose - local_goal_pose) > self.Lt:
            self.local_goal_pose, _ = self.lookahead_point(robot_pose)
        else:
            self.local_goal_pose = self.goal_pose

        self.get_logger().info("New local goal: (x,y) : " + str(self.local_goal_pose))
        
    
    def lookahead_point(self, robot_pose):
        """
        Find the goal point on the path to the robot

        Return:   
        goal_point: goal point on the path
        goal_index: index of the goal point on the path to the robot
        """
        index = self.pind
        global_path = np.array(self.global_path)
        robot_pose = np.array(robot_pose[:2])

        while (index + 1) < len(global_path):
            distance = np.linalg.norm(global_path[index] - robot_pose)
            if distance > self.Lt:
                break
            index += 1
        if self.pind <= index:
            self.pind = index
        goal_point = global_path[self.pind]
        return goal_point, index


    def get_random_goal(self, index):
        """
        """
        if self.episode < self.starting_episodes/2 or self.episode % 25 == 0:
            x = 0.65
            y = 0.05
        else:
            x = random.randrange(-25, 10) / 10.0
            y = random.randrange(-18, 18) / 10.0

        x += self.poses[index][0]
        y += self.poses[index][1]

        self.goal_pose = [x, y]


    def pause(self):
        self.is_paused = True

        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        #self.pause_physics_client.call_async(req) 
        success = None
        while success is None:
            future = self.pause_physics_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                success = future.result()
                self.get_logger().debug(f'Result of pausing Gazebo {success}')
            time.sleep(0.5)


    def unpause(self):
        self.is_paused = False

        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        #self.unpause_physics_client.call_async(req)
        success = None
        while success is None:
            future = self.unpause_physics_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                success = future.result()
                self.get_logger().debug(f'Result of unpausing Gazebo {success}')
            time.sleep(0.5)

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
    
    def create_clients(self,):
        # create reset world client 
        self.reset_world_client = self.create_client(Empty, 'reset_simulation')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')
        self.set_entity_state_client = self.create_client(SetEntityState, 'test/set_entity_state')
