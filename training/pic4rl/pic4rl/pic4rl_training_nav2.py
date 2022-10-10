#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

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
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from pic4rl.pic4rl_environment_nav2 import Pic4rlEnvironmentAPPLR
from ament_index_python.packages import get_package_share_directory

#from rclpy.executors import SingleThreadedExecutor
#from rclpy.executors import ExternalShutdownException


class Pic4rlTraining_APPLR(Pic4rlEnvironmentAPPLR):
    def __init__(self):
        super().__init__()
        rclpy.logging.set_logger_level('pic4rl_training', 10)

        trainer_params = os.path.join(get_package_share_directory('pic4rl'), 'config')
        configFilepath = os.path.join(trainer_params, 'main_param.yaml')
        with open(configFilepath, 'r') as file:
            configParams = yaml.safe_load(file)['main_node']['ros__parameters']

        self.declare_parameters(namespace='',
        parameters=[
            ('policy', configParams['policy']),
            ('policy_trainer', configParams['policy_trainer']),
            ('trainer_params', configParams['trainer_params']),
            ('max_lin_vel', configParams['max_lin_vel']),
            ('min_lin_vel', configParams['min_lin_vel']),
            ('max_ang_vel', configParams['max_ang_vel']),
            ('min_ang_vel', -configParams['min_ang_vel']),
            ])

        qos = QoSProfile(depth=10)

        self.train_policy = self.get_parameter('policy').get_parameter_value().string_value
        self.policy_trainer = self.get_parameter('policy_trainer').get_parameter_value().string_value
        self.training_params = self.get_parameter('trainer_params').get_parameter_value().string_value
        self.training_params = os.path.join(trainer_params, self.training_params)
        self.min_ang_vel = self.get_parameter('min_ang_vel').get_parameter_value().double_value
        self.min_lin_vel = self.get_parameter('min_lin_vel').get_parameter_value().double_value
        self.max_ang_vel = self.get_parameter('max_ang_vel').get_parameter_value().double_value
        self.max_lin_vel = self.get_parameter('max_lin_vel').get_parameter_value().double_value

        self.set_parser_list()
        self.trainer = self.instantiate_agent()
        self.get_logger().info('Starting process...')

    def instantiate_agent(self):
        """
        ACTION AND OBSERVATION SPACES settings
        """
        action=[
        [self.min_lin_vel, self.max_lin_vel], # max vel_x
        [self.min_ang_vel, self.max_ang_vel], # max vel_theta
        [5, 30], # vx_samples
        [10, 30], #v_theta samples
        [0.008, 0.08], # base obstacle scale
        [10, 40], # path dist scale
        [10, 40], # goal dist scale
        [0.3, 0.7], # inflation radius
        ]

        low_action = []
        high_action = []
        for i in range(len(action)):
            low_action.append(action[i][0])
            high_action.append(action[i][1])

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            shape=(len(action),),
            dtype=np.float32
        )
        self.get_logger().info('action space shape: {}'.format(self.action_space.high.size))

        state = []
        for i in range(self.lidar_points):
            state = state + [[0., 4.]]

        state = state + [[-math.pi, math.pi]] # goal angle or yaw

        state = state + [
        [self.min_lin_vel, self.max_lin_vel], # max vel_x
        [self.min_ang_vel, self.max_ang_vel], # max vel_theta
        [5, 30], # vx_samples
        [10, 30], #v_theta samples
        [0.008, 0.08], # base obstacle scale
        [10, 40], # path dist scale
        [10, 40], # goal dist scale
        [0.3, 0.7], # inflation radius
        ]

        if len(state)>0:
            low_state = []
            high_state = []
            for i in range(len(state)):
                low_state.append(state[i][0])
                high_state.append(state[i][1])

            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.get_logger().info('observation space size: {}'.format(self.observation_space.shape))
        
        # OFF-POLICY ALGORITHMS
        if self.policy_trainer == 'off-policy':
            parser = Trainer.get_argument()
            if self.train_policy == 'DDPG':
                self.get_logger().debug('Parsing DDPG parameters...')
                parser = DDPG.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = DDPG(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor = 3e-4,
                    lr_critic = 3e-4,
                    actor_units = (256, 128, 128),
                    critic_units = (256, 128, 128),
                    subclassing=False,
                    sigma = 0.2,
                    tau = 0.01,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05)
                self.get_logger().info('Instanciate DDPG agent...')

            if self.train_policy == 'TD3':
                self.get_logger().debug('Parsing TD3 parameters...')
                parser = TD3.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = TD3(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor = 3e-4,
                    lr_critic = 3e-4,
                    sigma = 0.2,
                    tau = 0.01,
                    epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    actor_update_freq = 2,
                    policy_noise = 0.2,
                    noise_clip = 0.5,
                    actor_units = (256, 128, 128),
                    critic_units = (256, 128, 128))
                self.get_logger().info('Instanciate TD3 agent...')
            
            if self.train_policy == 'SAC':
                self.get_logger().debug('Parsing SAC parameters...')
                parser = SAC.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = SAC(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action = self.action_space.high,
                    min_action=self.action_space.low,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    tau=5e-3,
                    alpha=.2,
                    auto_alpha=False, 
                    init_temperature=None,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    epsilon = 1.0, epsilon_decay = 0.996, epsilon_min = 0.05)
                self.get_logger().info('Instanciate SAC agent...')

            # Instanciate Policy Trainer
            trainer = Trainer(policy, self, args, test_env=None)

        # ON-POLICY ALGORITHMS
        if self.policy_trainer == 'on-policy':
            parser = OnPolicyTrainer.get_argument()
            
            if self.train_policy == 'PPO':
                self.get_logger().debug('Parsing PPO parameters...')
                parser = PPO.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = PPO(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    is_discrete = False,
                    max_action=self.action_space.high,
                    lr_actor = 1e-3,
                    lr_critic = 3e-3,
                    actor_units = (256, 256),
                    critic_units = (256, 256),
                    hidden_activation_actor="relu",
                    hidden_activation_critic="relu",
                    clip = True,
                    clip_ratio = 0.2,
                    horizon = self.param_dict["training_params"]["--horizon"],
                    enable_gae = self.param_dict["training_params"]["--enable-gae"],
                    normalize_adv = self.param_dict["training_params"]["--normalize-adv"],
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"])
                self.get_logger().info('Instanciate PPO agent...')
            # Instanciate Policy Trainer
            trainer = OnPolicyTrainer(policy, self, args, test_env=None)

        return trainer

    def set_parser_list(self):
        with open(self.training_params, 'r') as f:
            self.param_dict = yaml.load(f)

        self.parser_list = []
        for k,v in self.param_dict['training_params'].items():
            if v is not None:
                kv = k+'='+str(v)
                self.parser_list.append(kv)
            else:
                self.parser_list.append(k)

    def threadFunc(self):
        try:
            self.trainer()
        except Exception as e:
            self.get_logger().error("Error in starting trainer: {}".format(e))
            return
