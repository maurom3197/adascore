#!/usr/bin/env python3

import os
import tensorflow as tf
import traceback

import json
import numpy as np
import random
import yaml
import sys
import time
import math

import rclpy
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

import gym
from gym import spaces

from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.td3 import TD3
from tf2rl.algos.sac import SAC
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.tester_real import TesterReal
from adascore.tasks.SocialForceWindowReal.adascore_environment_social_cotroller import (
    Pic4rlEnvironmentAdascore,
)
from ament_index_python.packages import get_package_share_directory


class SocialForceWindowAdascore(Pic4rlEnvironmentAdascore):
    def __init__(self):
        super().__init__()
        self.log_check()
        train_params = self.parameters_declaration()

        self.set_parser_list(train_params)
        self.tester = self.instantiate_agent()


    def instantiate_agent(self):
        """
        ACTION AND OBSERVATION SPACES settings
        """
        action = [
            [0.5, 3.0],  # social_weight
            [0.5, 3.0],  # costmap_weight
            [0.1, 1.0],  # velocity_weight
            [0.1, 1.0],  # angle_weight
            [0.1, 1.5],  # distance_weight
            # [1.0,2.5], # wp_tolerance
            # [1.5,3.5] # sim_time
        ]

        low_action = []
        high_action = []
        for i in range(len(action)):
            low_action.append(action[i][0])
            high_action.append(action[i][1])

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(len(action),), dtype=np.float32
        )
        self.get_logger().info(
            "action space shape: {}".format(self.action_space.high.size)
        )
        self.get_logger().info(
            "action space max values: {}".format(self.action_space.high)
        )
        self.get_logger().info(
            "action space min values: {}".format(self.action_space.low)
        )

        state = []
        # Goal Info [angle, distance]
        state = state + [
            [-math.pi, math.pi],  # goal angle or yaw
            [0.0, 15.0],  # distance
        ]

        # Controller params at time t-1
        state = state + [
            [0.5, 3.0],  # social_weight
            [0.5, 3.0],  # costmap_weight
            [0.1, 1.0],  # velocity_weight
            [0.1, 1.0],  # angle_weight
            [0.1, 1.5],  # distance_weight
            # [1.0,2.5], # wp_tolerance
            # [1.5,3.5] # sim_time
        ]

        # Add people state
        for i in range(self.k_people):
            state = state + [
                [0.0, 5.0],  # distance
                [-math.pi, math.pi],  # angle
                [0.0, 1.5],  # velocity module
                [-math.pi, math.pi],  # yaw
            ]

        # Add LiDAR measures
        for i in range(self.lidar_points):
            state = state + [[0.0, 3.0]]

        if len(state) > 0:
            low_state = []
            high_state = []
            for i in range(len(state)):
                low_state.append(state[i][0])
                high_state.append(state[i][1])

            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.get_logger().info(
            "observation space size: {}".format(self.observation_space.shape)
        )

        # Set Epsilon-greedy starting value for exploration policy (minimum 0.05)
        epsilon = 0.0

        self.print_log()

        self.epsilon = 0.0

        # OFF-POLICY ALGORITHMS
        if self.policy_trainer == "off-policy":
            parser = TesterReal.get_argument()
            if self.train_policy == "DDPG":
                self.get_logger().debug("Parsing DDPG parameters...")
                parser = DDPG.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = DDPG(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor=2e-4,
                    lr_critic=2e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    subclassing=False,
                    sigma=0.2,
                    tau=0.01,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon=self.epsilon,
                    epsilon_decay=0.998,
                    epsilon_min=0.05,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate DDPG agent...")

            if self.train_policy == "TD3":
                self.get_logger().debug("Parsing TD3 parameters...")
                parser = TD3.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = TD3(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor=3e-4,
                    lr_critic=3e-4,
                    sigma=0.2,
                    tau=0.01,
                    epsilon=self.epsilon,
                    epsilon_decay=0.998,
                    epsilon_min=0.05,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    actor_update_freq=2,
                    policy_noise=0.2,
                    noise_clip=0.5,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate TD3 agent...")

            if self.train_policy == "SAC":
                self.get_logger().debug("Parsing SAC parameters...")
                parser = SAC.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = SAC(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    tau=5e-3,
                    alpha=0.2,
                    auto_alpha=False,
                    init_temperature=None,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon=self.epsilon,
                    epsilon_decay=0.998,
                    epsilon_min=0.05,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate SAC agent...")

            tester = TesterReal(policy, self, args, test_env=None)

        # ON-POLICY ALGORITHM TRAINER
        if self.policy_trainer == "on-policy":
            parser = TesterReal.get_argument()

            if self.train_policy == "PPO":
                self.get_logger().debug("Parsing PPO parameters...")
                parser = PPO.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = PPO(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    is_discrete=False,
                    max_action=self.action_space.high,
                    lr_actor=1e-3,
                    lr_critic=3e-3,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    hidden_activation_actor="relu",
                    hidden_activation_critic="relu",
                    clip=True,
                    clip_ratio=0.2,
                    horizon=self.horizon,
                    enable_gae=self.enable_gae,
                    normalize_adv=self.normalize_adv,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate PPO agent...")

            tester = TesterReal(policy, self, args, test_env=None)

        return tester

    def set_parser_list(self, params):
        """ """
        self.parser_list = []
        for k, v in params.items():
            if v is not None:
                kv = k + "="
                if k == "--logdir":
                    kv += self.logdir
                    self.get_logger().info(f"logdir set to: {kv}")
                elif k == "--model-dir":
                    kv += self.model_path
                    self.get_logger().info(f"model path set to: {kv}")
                elif k == "--rb-path-save":
                    kv += self.logdir + "/" + v
                    self.get_logger().info(f"rb path save set to: {kv}")
                elif k == "--rb-path-load":
                    kv += self.rb_path_load
                    self.get_logger().info(f"rb path load set to: {kv}")
                else:
                    kv += str(v)
                self.parser_list.append(kv)
            else:
                self.parser_list.append(k)

    def threadFunc(self):
        try:
            self.tester()
        except Exception:
            self.get_logger().error(
                f"Error in starting tester:\n {traceback.format_exc()}"
            )
            return

    def log_check(self):
        """
        Select the ROS2 log level.
        """
        try:
            self.log_level = int(os.environ["LOG_LEVEL"])
        except:
            self.log_level = 20
            self.get_logger().info("LOG_LEVEL not defined, setting default: INFO")

        self.get_logger().set_level(self.log_level)

    def print_log(self):
        """ """
        for i in range(len(self.log_dict)):
            self.get_logger().info(
                f"{list(self.log_dict)[i]}: {self.log_dict[list(self.log_dict)[i]]}"
            )

        self.get_logger().info(f"action space shape: {self.action_space.high.size}")
        self.get_logger().info(
            f"observation space size: {self.observation_space.high.size}\n"
        )

    def parameters_declaration(self):
        """ """
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )

        train_params_path = (
            self.get_parameter("training_params_path")
            .get_parameter_value()
            .string_value
        )

        with open(train_params_path, "r") as train_param_file:
            train_params = yaml.safe_load(train_param_file)["training_params"]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("gpu", train_params["--gpu"]),
                ("batch_size", train_params["--batch-size"]),
                ("n_warmup", train_params["--n-warmup"]),
            ],
        )

        self.train_policy = train_params["--policy"]
        self.policy_trainer = train_params["--policy_trainer"]
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )
        self.gpu = train_params["--gpu"]
        self.batch_size = train_params["--batch-size"]
        self.n_warmup = train_params["--n-warmup"]

        if self.train_policy == "PPO":
            self.horizon = int(train_params["--horizon"])
            self.normalize_adv = bool(train_params["--normalize-adv"])
            self.enable_gae = bool(train_params["--enable-gae"])
        else:
            self.memory_capacity = int(train_params["--memory-capacity"])

        self.log_dict = {
            "policy": train_params["--policy"],
            "max_steps": train_params["--max-steps"],
            "max_episode_steps": train_params["--episode-max-steps"],
            "sensor": self.sensor_type,
            "gpu": train_params["--gpu"],
        }

        return train_params
