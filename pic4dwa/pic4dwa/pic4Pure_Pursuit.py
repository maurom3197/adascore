import os
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from pic4utils.pic4utils import log_check, quat_to_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class Pic4Pure_Pursuit(Node):

	# Initialization of the class
	def __init__(self):
		super().__init__('pic4pure_pursuit')

		log_check(self)

		# Initialize variables
		qos 			= QoSProfile(depth=10)
		odom_topic      = '/odom'
		path_topic		= '/path'
		cmd_vel_topic	= '/cmd_vel'
		point_topic		= '/point'

		# Initilaize timer
		self.process_timer  = self.create_timer(
								0.05,
								self.process)

		# Initilaize subscribers
		self.Odom_sub       = self.create_subscription(
			Odometry,
			odom_topic,
			self.odom_callback,
			qos_profile=qos_profile_sensor_data
			)

		self.path_sub      = self.create_subscription(
			Path,
			path_topic,
			self.path_callback,
			qos_profile=qos
			)

		# Initialize publisher
		self.cmd_vel_pub 	= self.create_publisher(
			Twist,
			cmd_vel_topic,
			qos
			)

		self.point_pub 	= self.create_publisher(
			PointStamped,
			point_topic,
			qos
			)

		self.velocities = [0., 0.]
		self.max_v = 0.8
		self.max_yaw = 1.5
		self.Kp = 1
		self.K = 1.3
		self.trajectory = []
		self.pose_info = []

		self.get_logger().info("Initializing process")

	def process(self):
		"""
		"""
		if self.trajectory and self.pose_info:
			dist = []
			for i in self.trajectory:
				dist.append(np.hypot(self.pose_info[0] - i[0], self.pose_info[1] - i[1]))

			next_ind = np.argmin(dist) +2

			v = self.PID(self.velocities[0], self.max_v)
			theta = math.atan2(
				self.trajectory[next_ind][1] - self.pose_info[1],
				self.trajectory[next_ind][0] - self.pose_info[0]
				) 
			delta_theta = theta - self.pose_info[2]

			yaw = np.sign(delta_theta) * min(abs(self.K*delta_theta), self.max_yaw)

			self.send_action(v, yaw, next_ind)

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

	def path_callback(self, msg):
		"""
		"""
		trajectory = []
		for pose in msg.poses:
			trajectory.append([pose.pose.position.x, pose.pose.position.y])

		self.trajectory = trajectory

	def PID(self, current, target):
		return self.Kp * (target - current)

	def send_action(self, v, yaw, next_ind):
		"""
		"""
		twist = Twist()
		twist.linear.x = v
		twist.angular.z = yaw
		self.cmd_vel_pub.publish(twist)

		point = PointStamped()
		point.header.stamp       = Node.get_clock(self).now().to_msg()
		point.header.frame_id    = 'odom'
		point.point.x = self.trajectory[next_ind][0]
		point.point.y = self.trajectory[next_ind][1]
		self.point_pub.publish(point)

def main(args=None):
	rclpy.init()
	pic4pure_pursuit = Pic4Pure_Pursuit()
	
	try:
		rclpy.spin(pic4pure_pursuit)
	except KeyboardInterrupt:
		pic4pure_pursuit.get_logger().info("Shutting down")

	pic4pure_pursuit.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()		