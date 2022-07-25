import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu

from cv_bridge import CvBridge

from pic4rl.sensors import *

import os
import yaml

class Sensors():
    def __init__(self, node):
        #super().__init__("generic_sensors_node")
        self.param = self.get_param()

        self.odom_data = None
        self.laser_data = None
        self.depth_data = None
        self.rgb_data = None
        self.imu_data = None

        self.imu_sub = None
        self.depth_sub = None
        self.rgb_sub = None
        self.laser_sub = None
        self.odom_sub = None

        self.imu_process = None
        self.depth_process = None
        self.rgb_process = None
        self.laser_process = None
        self.odom_process = None
        self.node = node
        self.sensors = self.activate_sensors()
        self.bridge = CvBridge()
        

    def get_param(self):
        configFilepath = os.path.join(
            get_package_share_directory("pic4rl"), 'config',
            'main_param.yaml')
                            
        # Load the topic parameters
        with open(configFilepath, 'r') as file:
            configParams = yaml.safe_load(file)['main_node']['ros__parameters']

        return configParams

    def activate_sensors(self):
        if self.param["imu_enabled"]=="true":
            self.node.get_logger().info(self.param["imu_enabled"]+' subscription')
            self.imu_sub = self.node.create_subscription(
			Imu,
			self.param["sensors_topic"]["imu_topic"], 
			self.imu_cb,
			10)

            self.imu_process = ImuSensor()

        if self.param["camera_enabled"]=="true":
            self.node.get_logger().info('Depth subscription')
            self.depth_sub = self.node.create_subscription(
			    Image,
			    self.param["sensors_topic"]["depth_topic"], 
			    self.depth_camera_cb,
			    10)
            self.depth_process = DepthCamera(self.param["depth_param"]["width"], self.param["depth_param"]["height"], self.param["depth_param"]["dist_cutoff"],self.param["depth_param"]["show_image"])

            self.node.get_logger().info('RGB subscription')
            self.rgb_sub = self.node.create_subscription(
			    Image,
			    self.param["sensors_topic"]["rgb_topic"], 
			    self.rgb_camera_cb,
			    10)
            self.rgb_process = RGBCamera(self.param["rgb_param"]["width"], self.param["rgb_param"]["height"], self.param["rgb_param"]["show_image"])
            
        if self.param["lidar_enabled"]=="true":
            self.node.get_logger().debug('Laser subscription')
            self.laser_sub = self.node.create_subscription(
			LaserScan,
			self.param["sensors_topic"]["laser_topic"], 
			self.laser_scan_cb,
			10)
            self.laser_process = LaserScanSensor(
                self.param["laser_param"]["max_distance"], 
                self.param["laser_param"]["num_points"], 
                self.param["robot_type"],
                self.param["robot_radius"],
                self.param["robot_size"],
                self.param["collision_tollerance"]
                )
        
        self.node.get_logger().info('Odometry subscription')
        self.odom_sub = self.node.create_subscription(
			Odometry,
			self.param["sensors_topic"]["odom_topic"], 
			self.odometry_cb,
			10)
        self.odom_process = OdomSensor()

    def imu_cb(self, msg):
        self.imu_data = msg
        
    def depth_camera_cb(self, msg):
        self.depth_data = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        
    def rgb_camera_cb(self, msg):
        self.rgb_data = self.bridge.imgmsg_to_cv2(msg)

    def laser_scan_cb(self, msg):
        self.laser_data = msg.ranges
        
    def odometry_cb(self, msg):
        self.odom_data = msg

    def get_odom(self):
        if self.odom_sub is None:
            self.node.get_logger().warn('NO Odometry subscription')
            return None
        if self.odom_data is None:
            self.node.get_logger().warn('NO Odometry data')
            return None

        data = self.odom_process.process_data(self.odom_data)
        return data

    def get_depth(self):
        if self.depth_sub is None:
            self.get_logger().warn('NO Depth subscription')
            return None
        if self.depth_data is None:
            self.node.get_logger().warn('NO depth image')
            return None
        data = self.depth_process.process_data(self.depth_data)
        return data

    def get_rgb(self):
        if self.rgb_sub is None:
            self.get_logger().warn('NO RGB subscription')
            return None
        if self.rgb_data is None:
            self.node.get_logger().warn('NO RGB image')
            return None

        data = self.rgb_process.process_data(self.rgb_data)
        return data
        
    def get_imu(self):
        if self.imu_sub is None:
            self.get_logger().warn('NO IMU subscription')
            return None
        elif self.imu_data is None:
            self.get_logger().warn('NO IMU data')
        else:
            data = self.imu_process.process_data(self.imu_data)
            return data

    def get_laser(self):
        if self.laser_sub is None:
            self.node.get_logger().warn('NO laser subscription')
            return None, None, None
        if self.laser_data is None:
            self.node.get_logger().warn('NO laser data')
            return None, None, None

        data, min_obstacle_distance, collision = self.laser_process.process_data(self.laser_data)
        return data, min_obstacle_distance, collision
