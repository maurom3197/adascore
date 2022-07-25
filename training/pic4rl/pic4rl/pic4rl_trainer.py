#!/usr/bin/env python3

import yaml
import rclpy
import threading
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from pic4rl.pic4rl_training_lidar import Pic4rlTraining_Lidar
from pic4rl.pic4rl_training_camera import Pic4rlTraining_Camera
from pic4rl.pic4rl_training_applr import Pic4rlTraining_APPLR
from ament_index_python.packages import get_package_share_directory
from pic4rl.nav_param_client import DWBparamsClient

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
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

def main(args=None):
    """
    """
    rclpy.init()

    dwb_client = DWBparamsClient()
    #client = threading.Thread(target=dwb_client)
    #client.start()

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
        pic4rl_training= Pic4rlTraining_APPLR(dwb_client)
    
    print('instanciating executor... ')
    executor = MultiThreadedExecutor()
    print('executor done... ')
    executor.add_node(dwb_client)
    print('add client node... ')
    executor.add_node(pic4rl_training)
    print('add training node... ')

    th = threading.Thread(target=pic4rl_training.threadFunc)    
    th.start()
    
    try:
        print('try executor spin... ')
        executor.spin()
        print('spin done... ')
    finally:
            executor.shutdown()
            pic4rl_training.destroy_node()
            th.join()
            dwb_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()