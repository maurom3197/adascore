#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import yaml
import rclpy
import threading
#from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from pic4rl.pic4rl_training_lidar import Pic4rlTraining_Lidar
from pic4rl.pic4rl_training_camera import Pic4rlTraining_Camera
from pic4rl.pic4rl_training_nav2 import Pic4rlTraining_APPLR
from ament_index_python.packages import get_package_share_directory


def main(args=None):
    """
    """
    rclpy.init()

    configFilepath = os.path.join(
        get_package_share_directory("pic4rl"), 'config',
        'main_param.yaml'
    )

    with open(configFilepath, 'r') as file:
        configParams = yaml.safe_load(file)['main_node']['ros__parameters']

    if configParams['sensor'] == 'lidar':
        pic4rl_training= Pic4rlTraining_Lidar()
    elif configParams['sensor'] == 'camera':
        pic4rl_training= Pic4rlTraining_Camera()
    elif configParams['sensor'] == 'applr':
        pic4rl_training= Pic4rlTraining_APPLR()

    pic4rl_training.threadFunc()

    pic4rl_training.destroy_node()
    rclpy.shutdown()

    # th = threading.Thread(target=pic4rl_training.threadFunc)    
    # th.start()
    
    # try:
    #     rclpy.spin(pic4rl_training)
    # except:
    #     pic4rl_training.destroy_node()
    #     th.join()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()