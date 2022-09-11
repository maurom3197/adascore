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
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class Pic4Trajectory_Publisher(Node):

	# Initialization of the class
	def __init__(self):
		super().__init__('pic4trajectory_publisher')

		log_check(self)

		# Initialize variables
		qos 			= QoSProfile(depth=10)
		path_topic		= '/path'

		# Initilaize timer
		self.process_timer  = self.create_timer(
								1,
								self.process)
		# Initialize publisher
		self.path_pub = self.create_publisher(
			Path,
			path_topic,
			qos
			)

		cx = np.arange(0, 100, 0.5)
		cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

		self.trajectory = []
		for i in range(len(cx)):
			pose = PoseStamped()
			pose.pose.position.x = cx[i]
			pose.pose.position.y = cy[i]
			pose.pose.orientation.w = 1.0
			self.trajectory.append(pose)


		self.get_logger().info("Initializing process")

	def process(self):
		"""
		"""
		path = Path()
		path.header.stamp       = Node.get_clock(self).now().to_msg()
		path.header.frame_id    = 'odom'
		path.poses              = self.trajectory
		self.path_pub.publish(path)


def main(args=None):
	rclpy.init()
	pic4trajectory_publisher = Pic4Trajectory_Publisher()
	
	try:
		rclpy.spin(pic4trajectory_publisher)
	except KeyboardInterrupt:
		pic4trajectory_publisher.stop()
		pic4trajectory_publisher.get_logger().info("Shutting down")

	pic4trajectory_publisher.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()		