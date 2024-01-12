#!/usr/bin/env python3
import time
import subprocess
import math
import numpy as np
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav2_msgs.srv import ManageLifecycleNodes
import rclpy

def check_navigation(navigator):
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Goal succeeded!')
    elif result == TaskResult.CANCELED:
        print('Goal was canceled!')
    elif result == TaskResult.FAILED:
        print('Goal failed!')
    elif result == TaskResult.UNKNOWN:
        print('Navigation Result UNKNOWN!')
    return result

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

def filter_people(k_, distances, people_state_, people_info_):
    """
    """
    # Keep people state shape at (k,s_p), if people < k
    if len(people_state_) < k_:
        for i in range(k_-len(people_state_)):
            distances.append(20.0)
            people_state_.append([20.0, 0.0, 0.0, 0.0])
            people_info_.append([20.0, 20.0, 0.0, 0.0, 0.0, 0.0])

    distances = np.asarray(distances)
    min_people_distance = np.min(distances)
    people_state_ = np.asarray(people_state_)
    people_info_ = np.asarray(people_info_)

    # Filter the k closest people to the robot
    idx = np.argpartition(distances.ravel(), k_-1)
    #print('MIN distances idx: ', idx)
    people_state = people_state_[idx[:k_]] # shape (k,s_p)
    people_info = people_info_[idx[:k_]]

    #print('people state info (distance, angle, v_module, yaw) '+str(people_state))
    return people_state, people_info, min_people_distance

def lifecyclePause(navigator: BasicNavigator):
        """Pause nav2 lifecycle system."""
        navigator.info('Pause lifecycle nodes based on lifecycle_manager.')
        for srv_name, srv_type in navigator.get_service_names_and_types():
            if srv_type[0] == 'nav2_msgs/srv/ManageLifecycleNodes':
                navigator.info(f'Pause {srv_name}')
                mgr_client = navigator.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    navigator.info(f'{srv_name} service not available, waiting...')
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().PAUSE
                future = mgr_client.call_async(req)
                rclpy.spin_until_future_complete(navigator, future)
                future.result()
        return

def lifecycleResume(navigator: BasicNavigator):
        """Resume nav2 lifecycle system."""
        navigator.info('Resume lifecycle nodes based on lifecycle_manager.')
        for srv_name, srv_type in navigator.get_service_names_and_types():
            if srv_type[0] == 'nav2_msgs/srv/ManageLifecycleNodes':
                navigator.info(f'Resume {srv_name}')
                mgr_client = navigator.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    navigator.info(f'{srv_name} service not available, waiting...')
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().RESUME
                future = mgr_client.call_async(req)
                rclpy.spin_until_future_complete(navigator, future)
                future.result()
        return

def lifecycleReset(navigator: BasicNavigator):
        """Reset nav2 lifecycle system."""
        navigator.info('Reset lifecycle nodes based on lifecycle_manager.')
        for srv_name, srv_type in navigator.get_service_names_and_types():
            if srv_type[0] == 'nav2_msgs/srv/ManageLifecycleNodes':
                navigator.info(f'Reset {srv_name}')
                mgr_client = navigator.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    navigator.info(f'{srv_name} service not available, waiting...')
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().RESET
                future = mgr_client.call_async(req)
                rclpy.spin_until_future_complete(navigator, future)
                future.result()
        return