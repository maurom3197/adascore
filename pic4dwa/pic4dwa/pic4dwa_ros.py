import os
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from pic4utils.pic4utils import log_check, quat_to_euler, tf_decompose
from pic4dwa.pic4dwa import Pic4DWA
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan

class Pic4DWA_Ros(Node):

	# Initialization of the class
	def __init__(self):
		super().__init__('pic4dwa_ros')

		log_check(self)

		# Initialize variables
		qos 			= QoSProfile(depth=10)
		odom_topic      = '/odom'
		scan_topic		= '/scan'
		goal_topic		= '/goal_pose'
		cmd_vel_topic 	= '/cmd_vel'
		path_topic		= '/path'

		self.pose_info = [0.0, 0.0, 0.0]
		self.complete_pose = [0.0, 0.0, 0.0, 1.0]
		self.goal = [10.0, 10.0]
		self.ob = []
		self.u = [0.0, 0.0]
		self.max_lidar_range = 12.0
		self.goal_flag = True

		self.controller = Pic4DWA()

		# Initilaize timer
		self.process_timer  = self.create_timer(
								0.005,
								self.process)

		# Initilaize subscribers
		self.Odom_sub       = self.create_subscription(
								Odometry,
								odom_topic,
								self.odom_callback,
								qos_profile=qos_profile_sensor_data
								)

		self.lidar_sub      = self.create_subscription(
								LaserScan,
								scan_topic,
								self.scan_callback,
								qos_profile=qos_profile_sensor_data
								)

		self.goal_sub       = self.create_subscription(
								PoseStamped,
								goal_topic,
								self.goal_callback,
								qos_profile=qos_profile_sensor_data
								)

		# Initialize publisher
		self.cmd_vel_pub 	= self.create_publisher(
								Twist,
								cmd_vel_topic,
								qos
								)

		self.path_pub = self.create_publisher(
			Path,
			path_topic,
			qos
			)

		self.get_logger().info("Initializing process")

	def process(self):
		"""
		"""
		self.start = time.time()
		if self.ob and not self.goal_flag:
			# self.get_logger().info("Entering cycle ...")
			x, ob, goal = self.controller.get_env_data(self.pose_info, self.u, self.ob, self.goal)
			dw = self.controller.calc_dynamic_window(x)
			u, trajectory = self.controller.calc_control_and_trajectory(x, dw, goal, ob)

			# check reaching goal
			dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
			if dist_to_goal <= self.controller.goal_tollerance:
				self.stop()
				self.u = [0.0, 0.0]
				self.goal_flag = True
				self.get_logger().info("Goal!!")
			else:
				self.send_command(u, trajectory)

			self.ob = []

		# print(time.time()-self.start)

	def odom_callback(self, msg):
		"""
		"""
		xr 	= msg.pose.pose.position.x
		yr 	= msg.pose.pose.position.y
		zqr = msg.pose.pose.orientation.z
		wqr = msg.pose.pose.orientation.w

		vx = msg.twist.twist.linear.x
		wz = msg.twist.twist.angular.z

		zr 	= quat_to_euler(zqr, wqr)
		
		self.pose_info = [xr, yr, zr]
		self.complete_pose = [xr, yr, zqr, wqr]
		self.u = [vx, wz]

	def scan_callback(self, msg):
		"""
		"""
		lidar_measurements = msg.ranges
		ob = []
		for index, point in enumerate(lidar_measurements):
			# if point == float('inf'):
			# 	point = self.max_lidar_range
			p = [point*math.cos(index*0.0174533), point*math.sin(index*0.0174533)]
			p = tf_decompose(self.complete_pose ,[p[0], p[1], 0.0, 1.0])
			if math.isnan(p[0]) or math.isnan(p[1]):
				p = [float('inf'), float('inf'), 0., 1.]
			ob.append([p[0], p[1]])

		self.ob = ob

	def goal_callback(self, msg):
		"""
		"""
		xg = msg.pose.position.x
		yg = msg.pose.position.y

		self.goal = [xg, yg]
		self.goal_flag = False

	def send_command(self, u, trajectory):
		"""
		"""
		vel = Twist()
		vel.linear.x = u[0]
		vel.angular.z = u[1]
		self.cmd_vel_pub.publish(vel)

		t = []
		for i in trajectory:
			pose = PoseStamped()
			pose.pose.position.x = i[0]
			pose.pose.position.y = i[1]
			pose.pose.orientation.w = 1.0
			t.append(pose)

		path = Path()
		path.header.stamp       = Node.get_clock(self).now().to_msg()
		path.header.frame_id    = 'odom'
		path.poses              = t
		self.path_pub.publish(path)

	def stop(self):
		"""
		"""
		vel = Twist()
		vel.linear.x = 0.0
		vel.angular.z = 0.0
		for i in range (5):
			self.cmd_vel_pub.publish(vel)
			time.sleep(0.2)


def main(args=None):
	rclpy.init()
	pic4dwa_ros = Pic4DWA_Ros()
	
	try:
		rclpy.spin(pic4dwa_ros)
	except KeyboardInterrupt:
		pic4dwa_ros.stop()
		pic4dwa_ros.get_logger().info("Shutting down")

	pic4dwa_ros.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()		