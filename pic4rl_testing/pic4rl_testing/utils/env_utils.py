#!/usr/bin/env python3

# Python libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

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
from nav2_simple_commander.robot_navigator import BasicNavigator, NavigationResult


def frequency_control(params_update_freq):
    #print("Sleeping for: "+str(1/self.params_update_freq) +' s')
    time.sleep(1/params_update_freq)

def compute_frequency(t0):
        t1 = time.perf_counter()
        step_time = t1-t0
        t0 = t1
        action_hz = 1./(step_time)
        return action_hz, t1
        
def get_goals_and_poses(data_path):
    """
    """
    data = json.load(open(data_path,'r'))

    return data["initial_pose"], data["goals"], data["poses"], data["agents"]

def check_navigation(navigator):
    result = navigator.getResult()
    if result == NavigationResult.SUCCEEDED:
        print('Goal succeeded!')
    elif result == NavigationResult.CANCELED:
        print('Goal was canceled!')
    elif result == NavigationResult.FAILED:
        print('Goal failed!')
    elif result == NavigationResult.UNKNOWN:
        print('Navigation Result UNKNOWN!')
    return result

def filter_people(k_, distances, people_state_, people_info_):
    """
    """
    # Keep people state shape at (k,s_p), if people < k
    if len(people_state_) < k_:
        for i in range(k_-len(people_state_)):
            distances.append(20.0)
            people_state_.append([20.0, 0.0, 0.0, 0.0])
            people_info_.append([20.0, 20.0, 0.0, 0.0, 0.0])

    distances = np.array(distances)
    min_people_distance = np.min(distances)
    people_state_ = np.array(people_state_)
    people_info_ = np.array(people_info_)

    # Filter the k closest people to the robot
    idx = np.argpartition(distances.ravel(), k_-1)
    #print('MIN distances idx: ', idx)
    people_state = people_state_[idx[:k_]] # shape (k,s_p)
    people_info = people_info_[idx[:k_]]

    #print('people state info (distance, angle, v_module, yaw) '+str(people_state))
    return people_state, people_info, min_people_distance

def process_odom(goal_pose, odom):

    goal_distance = math.sqrt(
        (goal_pose[0]-odom[0])**2
        + (goal_pose[1]-odom[1])**2)

    path_theta = math.atan2(
        goal_pose[1]-odom[1],
        goal_pose[0]-odom[0])

    goal_angle = path_theta - odom[2]

    if goal_angle > math.pi:
        goal_angle -= 2 * math.pi

    elif goal_angle < -math.pi:
        goal_angle += 2 * math.pi

    goal_info = [goal_distance, goal_angle]
    robot_pose = [odom[0], odom[1], odom[2]]

    return goal_info, robot_pose

def process_people(person_position, robot_pose):
    """
    """
    distance = math.sqrt(
        (person_position[0]-robot_pose[0])**2
        + (person_position[1]-robot_pose[1])**2)

    path_theta = math.atan2(
        person_position[1]-robot_pose[1],
        person_position[0]-robot_pose[0])

    angle = path_theta - robot_pose[2]

    if angle > math.pi:
        angle -= 2 * math.pi

    elif angle < -math.pi:
        angle += 2 * math.pi

    return distance, angle

def create_logdir(policy, sensor, logdir):
    """
    """
    logdir_name = f"Social_nav_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{sensor}_{policy}/"
    logdir_f = Path(os.path.join(logdir, logdir_name))
    logdir_f.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logdir, logdir_name, 'screen_logger.log'), 
        level=logging.INFO)
    return logdir_name, logdir_f

def euler_from_quaternion(quat):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        return v, norm
        #norm=np.finfo(v.dtype).eps
    return v/norm, norm

def normalize_angle(theta):
    # theta should be in the range [-pi, pi]
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi
    return theta

def restart_gazebo(gazebo_client):

    print("Shutting down gazebo...")
    subprocess.run(
        "pkill -9 gzserver",
        shell=True,
        stdout=subprocess.DEVNULL
        )
    time.sleep(5.0)

    if gazebo_client:
        subprocess.run(
            "pkill -9 gzclient",
            shell=True,
            stdout=subprocess.DEVNULL
            )

        time.sleep(5.0)

    print("Launching gazebo...")
    subprocess.run(
        "ros2 launch hunav_gazebo_wrapper social_small_indoor.launch.py &",
        shell=True,
        stdout=subprocess.DEVNULL
        )

    time.sleep(20.0)
    
def restart_nav2():
    nav2_nodes = ["bt_navigator", "planner_server", 
    "controller_server", "map_server", 
    "recoveries_server", "lifecycle_manager_navigation"
     ]   

    for proc in nav2_nodes:
        print("Killing nav process: "+proc)
        subprocess.run(
            "killall -9 "+proc,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(5.0)

    print("launching nav2 again...")
    subprocess.run(
        "ros2 launch pic4nav social_nav_with_map.launch.py &",
        shell=True,
        stdout=subprocess.DEVNULL
        )
    time.sleep(20.0)

def plot_costmap(image):
        colormap = np.asarray(image*255, dtype = np.uint8)
        cv2.namedWindow('Local Costmap', cv2.WINDOW_NORMAL)
        cv2.imshow('Local Costmap',colormap)
        cv2.waitKey(1)