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

from geometry_msgs.msg import Pose, PoseStamped
from gazebo_msgs.srv import SetEntityState
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
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

from people_msgs.msg import People


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
        self.main_params_path = (
            self.get_parameter("main_params_path").get_parameter_value().string_value
        )
        training_params_path = (
            self.get_parameter("training_params_path")
            .get_parameter_value()
            .string_value
        )
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), "models/goal_box/model.sdf"
        )

        with open(training_params_path, "r") as train_param_file:
            training_params = yaml.safe_load(train_param_file)["training_params"]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("mode", rclpy.Parameter.Type.STRING),
                ("data_path", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("agents_config", rclpy.Parameter.Type.STRING),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("max_lin_vel", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
                ("laser_param.total_points", rclpy.Parameter.Type.INTEGER),
                ("sensor", rclpy.Parameter.Type.STRING),
                ("use_localization", rclpy.Parameter.Type.BOOL),
            ],
        )

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.data_path = (
            self.get_parameter("data_path").get_parameter_value().string_value
        )
        self.data_path = os.path.join(goals_path, self.data_path)
        print(training_params["--change_goal_and_pose"])
        self.change_episode = int(training_params["--change_goal_and_pose"])
        self.starting_episodes = int(training_params["--starting_episodes"])
        self.timeout_steps = int(training_params["--episode-max-steps"])
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.agents_config = (
            self.get_parameter("agents_config").get_parameter_value().string_value
        )
        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.max_lin_vel = (
            self.get_parameter("max_lin_vel").get_parameter_value().double_value
        )
        self.params_update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
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
        self.total_points = (
            self.get_parameter("laser_param.total_points")
            .get_parameter_value()
            .integer_value
        )
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )
        self.use_localization = (
            self.get_parameter("use_localization").get_parameter_value().bool_value
        )
        self.bag_process = None
        self.bag_episode = 0

        # create Sensor class to get and process sensor data
        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        # create local goal info and subscribe to local goal topic
        self.local_goal_pose = None

        self.goal_sub = self.create_subscription(
            PoseStamped, "goal_pose", self.goal_callback, qos
        )

        self.local_goal_sub = self.create_subscription(
            PoseStamped, "local_goal", self.local_goal_callback, qos
        )

        log_path = os.path.join(
            get_package_share_directory(self.package_name),
            "../../../../",
            training_params["--logdir"],
        )

        # create log dir
        self.logdir = create_logdir(
            training_params["--policy"], self.sensor_type, log_path
        )
        self.get_logger().info(f"Logdir: {self.logdir}")

        if "--model-dir" in training_params:
            self.model_path = os.path.join(
                get_package_share_directory(self.package_name),
                "../../../../",
                training_params["--model-dir"],
            )

        if "--rb-path-load" in training_params:
            self.rb_path_load = os.path.join(
                get_package_share_directory(self.package_name),
                "../../../../",
                training_params["--rb-path-load"],
            )

        self.create_clients()
        self.spin_sensors_callbacks()

        # init weights publisher
        self.cost_weights_pub = self.create_publisher(
            Float32MultiArray, "cost_weights", qos
        )

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
        self.previous_local_goal_info = [0.0, 0.0]

        self.initial_pose, self.goals, self.poses, self.agents = (
            self.get_goals_and_poses()
        )
        self.goal_pose = None
        self.init_nav_params = [
            2.0,  # social_weight
            2.0,  # costmap_weight
            0.8,  # velocity_weight
            0.6,  # angle_weight
            1.0,  # distance_weight
        ]

        self.get_logger().info(
            "Navigation params update at: " + str(self.params_update_freq) + " Hz"
        )

    def step(self, action, episode_step=0):
        """ """
        self.get_logger().debug("Env step : " + str(episode_step))
        self.episode_step = episode_step

        self.get_logger().debug("Action received (nav2 params): " + str(action))
        params = action.tolist()

        observation, reward, done = self._step(params)
        info = None

        return observation, reward, done, info

    def _step(self, nav_params=None, reset_step=False):
        """ """
        self.get_logger().debug("sending action...")
        self.send_action(nav_params)

        self.spin_sensors_callbacks()
        self.get_logger().debug("getting sensor data...")
        (
            lidar_measurements,
            goal_info,
            local_goal_info,
            robot_pose,
            robot_velocity,
            collision,
        ) = self.get_sensor_data()
        self.get_logger().debug("getting people data...")

        people_state, people_info = self.get_people_state(robot_pose, robot_velocity)
        wr, wp = self.sfm.computeSocialWork()
        social_work = wr + wp

        if not reset_step:
            self.get_logger().debug("checking events...")
            done, event = self.check_events(
                lidar_measurements, goal_info, robot_pose, collision
            )
            reward = None
            self.get_logger().debug("getting observation...")
            observation = self.get_observation(
                lidar_measurements, goal_info, robot_pose, people_state, nav_params
            )
        else:
            reward = None
            observation = None
            done = False
            event = "None"

        self.update_state(
            lidar_measurements,
            local_goal_info,
            robot_pose,
            people_state,
            nav_params,
            done,
            event,
        )

        return observation, reward, done

    def spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning for sensor_msg...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            self.get_logger().debug("None in sensor_msg... spinning again...")
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        rclpy.spin_once(self)

    def get_goals_and_poses(self):
        """ """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"], data["agents"]

    def get_sensor_data(self):
        """ """
        sensor_data = {}
        sensor_data["scan"], min_obstacle_distance, collision = self.sensors.get_laser(
            min_obstacle_distance=True
        )
        sensor_data["odom"], sensor_data["velocity"] = self.sensors.get_odom(vel=True)

        if sensor_data["scan"] is None:
            self.get_logger().debug("scan data is None...")
            sensor_data["scan"] = np.squeeze(
                np.ones((1, self.lidar_points)) * 2.0
            ).tolist()
            min_obstacle_distance = 2.0
            collision = False
        if sensor_data["odom"] is None:
            self.get_logger().debug("odom data is None...")
            sensor_data["odom"] = [0.0, 0.0, 0.0]

        if self.goal_pose is None:
            self.get_logger().debug("goal data is None...")
            self.goal_pose = [0.0, 0.0]

        goal_info, robot_pose = process_odom(self.goal_pose, sensor_data["odom"])

        if self.local_goal_pose is None:
            self.get_logger().debug("local goal data is None...")
            self.local_goal_pose = [0.0, 0.0]

        local_goal_info, _ = process_odom(self.local_goal_pose, sensor_data["odom"])

        lidar_measurements = sensor_data["scan"]
        self.min_obstacle_distance = min_obstacle_distance
        velocity = sensor_data["velocity"]

        return (
            lidar_measurements,
            goal_info,
            local_goal_info,
            robot_pose,
            velocity,
            collision,
        )

    def send_action(self, params):
        """ """
        controller_params = params

        self.set_controller_params(controller_params)

        # for testing purposes
        self.cost_weights_pub.publish(Float32MultiArray(data=controller_params))

        # Regulate the step frequency of the environment
        action_hz, t1 = compute_frequency(self.t0)
        self.get_logger().debug(f"frequency : {action_hz}")
        self.t0 = t1
        if action_hz > self.params_update_freq:
            frequency_control(self.params_update_freq)
            self.get_logger().debug("Sending action at " + str(action_hz))

    def get_observation(
        self, lidar_measurements, goal_info, robot_pose, people_state, nav_params
    ):

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

        state = np.array(state_list, dtype=np.float32)
        return state

    def update_state(
        self,
        lidar_measurements,
        local_goal_info,
        robot_pose,
        people_state,
        nav_params,
        done,
        event,
    ):
        """ """
        self.previous_lidar_measurements = lidar_measurements
        self.previous_local_goal_info = local_goal_info
        self.previous_robot_pose = robot_pose
        self.people_state = people_state
        self.previous_nav_params = nav_params
        self.previous_event = event

    def reset(self, n_episode, evaluate=True):
        """ """

        self.episode = n_episode
        self.evaluate = evaluate

        nav_params = self.init_nav_params
        (
            _,
            _,
            _,
        ) = self._step(nav_params, reset_step=True)
        (
            observation,
            _,
            _,
        ) = self._step(nav_params)

        ## Wait until the robot receives the goal
        while (self.goal_pose[0] == 0) and (self.goal_pose[1] == 0):
            self.get_logger().debug("waiting for local goal...")
            rclpy.spin_once(self)
        return observation

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """
        Check if the episode is done or if an event has occurred
        """

        # check if we reached the goal
        if goal_info[0] < self.goal_tolerance:
            self.get_logger().debug("Goal reached")
            done = True
            event = "Goal reached"
            return done, event

        # check if the robot collided
        if collision:
            self.get_logger().debug("Collision")
            done = False
            event = "Collision"
            return done, event

        if self.episode_step == self.timeout_steps - 1:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            logging.info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            return True, "timeout"

        return False, "None"

    def get_people_state(self, robot_pose, robot_velocity):
        """ """
        # Spin once to get the people message
        rclpy.spin_once(self)

        people_state, people_info, min_people_distance = self.sfm.get_people(
            robot_pose, robot_velocity
        )
        self.min_people_distance = min_people_distance
        self.get_logger().debug("Min people distance: " + str(min_people_distance))

        return people_state, people_info

    def send_set_request_controller(self, param_values):
        self.set_req_controller.parameters = [
            Parameter(
                name="FollowPath.social_weight", value=param_values[0]
            ).to_parameter_msg(),
            Parameter(
                name="FollowPath.costmap_weight", value=param_values[1]
            ).to_parameter_msg(),
            Parameter(
                name="FollowPath.velocity_weight", value=param_values[2]
            ).to_parameter_msg(),
            Parameter(
                name="FollowPath.angle_weight", value=param_values[3]
            ).to_parameter_msg(),
            Parameter(
                name="FollowPath.distance_weight", value=param_values[4]
            ).to_parameter_msg(),
        ]
        future = self.set_cli_controller.call_async(self.set_req_controller)
        return future

    def set_controller_params(self, controller_params):
        self.get_logger().debug(
            "setting controller params to: " + str(controller_params)
        )
        self.set_req_controller = SetParameters.Request()
        future = self.send_set_request_controller(controller_params)
        rclpy.spin_until_future_complete(self, future)

        try:
            get_response = future.result()
            self.get_logger().debug("Result %s" % (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug("Service call failed %r" % (e,))

    def compute_frequency(
        self,
    ):
        t1 = time.perf_counter()
        step_time = t1 - self.t0
        self.t0 = t1
        action_hz = 1.0 / (step_time)
        self.get_logger().debug("Sending action at " + str(action_hz))

    def send_get_request_controller(self):
        self.get_req_controller.names = [
            "FollowPath.social_weight",
            "FollowPath.costmap_weight",
            "FollowPath.velocity_weight",
            "FollowPath.angle_weight",
            "FollowPath.distance_weight",
            #'FollowPath.wp_tolerance',
            #'FollowPath.sim_time'
        ]
        future = self.get_cli_controller.call_async(self.get_req_controller)
        return future

    def get_controller_params(
        self,
    ):
        future = self.send_get_request_controller()
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().info(
                "Result %s %s %s %s %s %s %s"
                % (
                    get_response.values[0].double_value,
                    get_response.values[1].double_value,
                    get_response.values[2].integer_value,
                    get_response.values[3].integer_value,
                    get_response.values[4].double_value,
                    # get_response.values[5].double_value, # only if wp_tolerance is used
                    # get_response.values[6].double_value  # only if sim_time is used
                )
            )

        except Exception as e:
            self.get_logger().info("Service call failed %r" % (e,))

    def create_clients(
        self,
    ):
        # # create Controller parameter client
        self.get_cli_controller = self.create_client(
            GetParameters, "/controller_server/get_parameters"
        )
        while not self.get_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.get_req_controller = GetParameters.Request()

        self.set_cli_controller = self.create_client(
            SetParameters, "/controller_server/set_parameters"
        )
        while not self.set_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.set_req_controller = SetParameters.Request()

    def local_goal_callback(self, msg):
        self.local_goal_pose = [msg.pose.position.x, msg.pose.position.y]

    def goal_callback(self, msg):
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]
