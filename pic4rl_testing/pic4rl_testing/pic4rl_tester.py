#!/usr/bin/env python3

import yaml
import rclpy
import threading
#from pic4rl_testing.pic4rl_testing_nav2 import Pic4rlTesting_APPLR
from pic4rl_testing.pic4rl_testing_social_nav import Pic4rlTesting_APPLR_people
from pic4rl_testing.pic4rl_testing_costmap import Pic4rlTesting_APPLR_costmap
from ament_index_python.packages import get_package_share_directory

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

def main(args=None):
    """
    """
    configFilepath = os.path.join(
        get_package_share_directory("pic4rl_testing"), 'config',
        'main_params.yaml'
    )

    with open(configFilepath, 'r') as file:
        configParams = yaml.safe_load(file)['main_node']['ros__parameters']

    rclpy.init()

    pic4rl_testing = Pic4rlTesting_APPLR_costmap()
    pic4rl_testing.get_logger().info(
                "Initialized Testing: APPLR agent, Task: social_nav with people\n\n")

    pic4rl_testing.threadFunc()

    pic4rl_testing.destroy_node()
    rclpy.shutdown()

    # th = threading.Thread(target=pic4rl_testing.threadFunc)    
    # th.start()
    
    # try:
    #     rclpy.spin(pic4rl_testing)
    # except:
    #     pic4rl_testing.destroy_node()
    #     th.join()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()