import os
import math
import numpy as np

class Pic4DWA:
	"""
	"""
	def __init__(self):
		"""
		"""
		# DWA PARAMETERS
		# Velocities
		self.max_speed = 0.5  # [m/s]
		self.min_speed = -0.2  # [m/s]
		self.max_yaw_rate = 30.0 * math.pi / 180.0  # [rad/s]
		# Accelerations
		self.max_accel = 0.2  # [m/ss]
		self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
		# Resolution
		self.v_resolution = 0.01  # [m/s]
		self.yaw_rate_resolution = 0.5 * math.pi / 180.0  # [rad/s]
		self.dt = 0.1  # [s] Time tick for motion prediction
		self.predict_time = 3.0  # [s]
		# Normalizers
		self.alfa   = 1/4				# heading
		self.beta   = 1/3 				# obstacle 
		self.gamma  = 1/self.max_speed	# speed

		self.robot_stuck_flag_cons = 0.01  # constant to prevent robot stucked
		self.robot_type = 'rectangle'
		self.min_cost = 1000
		self.goal_tollerance = 0.3
		self.R = self.max_speed / self.max_yaw_rate

		if self.robot_type == 'circle':
			self.robot_radius = 1.0  # [m] for collision check
		elif self.robot_type == 'rectangle':
			self.robot_width = 0.26  # [m] for collision check
			self.robot_length = 0.26 # [m] for collision check

		# self.x = [0.0, 0.0, math.pi/8.0, 0.0, 0.0] # state: [x,y,th,v,omega]
		# self.goal = [0.0, 0.0] # goal: [x,y]
		# self.trajectory = self.x
		# self.ob = []

	def get_env_data(
			self, 
			pose = [0.0, 0.0, 0.0], 
			u = [0.0, 0.0],
			ob = [], 
			goal = [0.0, 0.0]
			):
		"""
		"""
		return [pose[0], pose[1], pose[2], u[0], u[1]], ob, goal


	def motion(self, x, u):
		"""
		motion model
		"""
		y = [0, 0, 0, 0, 0]
		y[2] = x[2] + u[1] * self.dt
		y[0] = x[0] + u[0] * math.cos(x[2]) * self.dt
		y[1] = x[1] + u[0] * math.sin(x[2]) * self.dt
		y[3] = u[0]
		y[4] = u[1]

		return y


	def calc_dynamic_window(self, x):
		"""
		calculation dynamic window based on current state x
		"""
		# Dynamic window from robot specification
		Vs = [self.min_speed, self.max_speed,
			  -self.max_yaw_rate, self.max_yaw_rate]

		# Dynamic window from motion model
		Vd = [x[3] - self.max_accel * self.dt,
			  x[3] + self.max_accel * self.dt,
			  x[4] - self.max_delta_yaw_rate * self.dt,
			  x[4] + self.max_delta_yaw_rate * self.dt]

		#  [v_min, v_max, yaw_rate_min, yaw_rate_max]
		dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
			  max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

		return dw

	def calc_control_and_trajectory(self, x, dw, goal, ob):
		"""
		calculation final input with dynamic window
		"""
		min_cost = float("inf")
		min_heading_cost = float("inf")
		min_speed_cost = float("inf")
		min_ob_cost = float("inf")
		best_angle = float('inf')
		best_u = [0.0, 0.0]
		best_trajectory = [x]

		# evaluate all trajectory with sampled input in dynamic window
		for v in np.arange(dw[0], dw[1], self.v_resolution):
			for w in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

				trajectory = self.predict_trajectory(x, v, w)

				angle   = self.calc_heading_cost(trajectory, goal)
				dist    = self.calc_obstacle_cost(trajectory, ob)
				vel     = self.calc_vel_cost(trajectory)

				heading_cost    = self.alfa * angle
				ob_cost         = self.beta * dist
				speed_cost      = self.gamma * vel

				final_cost = heading_cost + speed_cost + ob_cost

				# search minimum trajectory
				if min_cost >= final_cost:
					min_cost = final_cost
					min_heading_cost = heading_cost
					min_speed_cost = speed_cost
					min_ob_cost = ob_cost
					best_u = [v, w]
					best_trajectory = trajectory
					best_angle = angle
					if abs(best_u[0]) < self.robot_stuck_flag_cons \
							and abs(x[3]) < self.robot_stuck_flag_cons:
						# to ensure the robot do not get stuck in
						# best v=0 m/s (in front of an obstacle) and
						# best omega=0 rad/s (heading to the goal with
						# angle difference of 0)
						best_u[1] = -self.max_delta_yaw_rate

		# print("Best_traj: ", best_trajectory)
		# print("Min_cost: ", min_cost)
		# print("Min_angle: ", best_angle)
		# print("min_heading_cost: ", min_heading_cost)
		# print("min_speed_cost: ", min_speed_cost)
		# print("min_ob_cost: ", min_ob_cost)
		# print("------------------------\n")

		return best_u, best_trajectory

	def predict_trajectory(self, x, v, w):
		"""
		predict trajectory with an input
		"""
		trajectory = [x]
		for i in range(int(self.predict_time/self.dt)):
			x = self.motion(x, [v, w])
			trajectory.append(x)

		return trajectory

	def calc_vel_cost(self, trajectory):
		"""
		"""
		return abs(self.max_speed - trajectory[-1][3])
		# return abs(self.max_speed - (trajectory[-1][3] + trajectory[-1][4]*self.R))

	def calc_obstacle_cost(self, trajectory, ob):
		"""
		calc obstacle cost inf: collision
		"""
		ob = np.array(ob)
		trajectory = np.array(trajectory)
		ox = ob[:, 0]
		oy = ob[:,1]
		dx = trajectory[:,0] - ox[:,None]
		dy = trajectory[:,1] - oy[:,None]
		r = np.hypot(dx, dy)

		if self.robot_type == 'rectangle':
			if np.array(r <= self.robot_width/2).any():
				return 100.
		elif self.robot_type == 'circle':
			if np.array(r <= self.robot_radius).any():
				return 100.

		min_r = np.min(r)
		return 1.0 / min_r  # OK

	def calc_heading_cost(self, trajectory, goal):
		"""
			calc to goal cost with angle difference
		"""
		# Using angle: worse performances
		# dx = goal[0] - trajectory[-1][0]
		# dy = goal[1] - trajectory[-1][1]
		# error_angle = math.atan2(dy, dx)
		# cost_angle = error_angle - trajectory[-1][2]
		# angle = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

		# Using distance from goal: best performances
		dx = trajectory[-1][0]
		dy = trajectory[-1][1]
		angle = np.hypot(dx,dy)

		return 1/angle


def main():
	"""
	"""
	dwa_controller = Pic4DWA()
	print('Starting process')
	while True:
		x, ob, goal = dwa_controller.get_env_data()
		dw = dwa_controller.calc_dynamic_window(x)
		u, trajectory = dwa_controller.calc_control_and_trajectory(x, dw, goal, ob)
		# x = dwa_controller.motion(x, u)  # simulate robot
		# trajectory = np.vstack((trajectory, x))  # store state history

		# check reaching goal
		dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
		if dist_to_goal <= config.robot_radius:
			print("Goal!!")
			break

	print("Done")