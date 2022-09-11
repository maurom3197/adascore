import os
import math
import numpy as np

class Pic4DWA_Omni:
	"""
	"""
	def __init__(self):
		"""
		"""
		# DWA PARAMETERS
		# Velocities
		self.max_speed_x = 0.5  # [m/s]
		self.min_speed_x = -0.5  # [m/s]
		self.max_speed_y = 0.5  # [m/s]
		self.min_speed_y = -0.5  # [m/s]
		self.max_yaw_rate = 0.0 # [rad/s]
		# Accelerations
		self.max_accel_x = 0.2  # [m/ss]
		self.max_accel_y = 0.2  # [m/ss]
		self.max_delta_yaw_rate = 0.0  # [rad/ss]
		# Resolution
		self.v_x_resolution = 0.01  # [m/s]
		self.v_y_resolution = 0.01  # [m/s]
		self.yaw_rate_resolution = 0.5 * math.pi / 180.0  # [rad/s]
		self.dt = 0.1  # [s] Time tick for motion prediction
		self.predict_time = 3.0  # [s]
		# Normalizers
		self.alfa   = 1/4				# heading
		self.beta   = 1/2.5				# obstacle 
		self.gamma  = 1/1.3				# speed

		self.robot_stuck_flag_cons = 0.01  # constant to prevent robot stucked
		self.robot_type = 'rectangle'
		self.min_cost = 1000
		self.goal_tollerance = 0.3
		# self.R = self.max_speed_x / self.max_yaw_rate

		self.robot_radius = 1.0  # [m] for collision check circle
		self.robot_width = 0.35  # [m] for collision check rectangle
		self.robot_length = 0.385 # [m] for collision check rectangle

		self.get_collision_vector()

		# self.x = [0.0, 0.0, math.pi/8.0, 0.0, 0.0] # state: [x,y,th,v,omega]
		# self.goal = [0.0, 0.0] # goal: [x,y]
		# self.trajectory = self.x
		# self.ob = []

		# Dynamic window from robot specification
		self.Vs = [
			self.min_speed_x, self.max_speed_x,
			self.min_speed_y, self.max_speed_y,
			-self.max_yaw_rate, self.max_yaw_rate
			]

	def get_collision_vector(self):
		"""
		"""
		if self.robot_type == "circle":
			self.collision_vector = np.full(360, self.robot_radius)

		elif self.robot_type == "rectangle":
			sL = self.robot_length/2 # Rover lenght semiaxis
			sW = self.robot_width/2 # Rover width semiaxis

			degrees = np.arange(0, math.pi*2, math.pi/180)
			vec1 = sL/np.cos(degrees)
			vec2 = sW/np.sin(degrees)

			self.collision_vector = np.minimum(np.abs(vec1), np.abs(vec2))

	def get_env_data(
			self, 
			pose = [0.0, 0.0, 0.0], 	# [m, m, rad] x, y, angle
			u = [0.0, 0.0, 0.0],		# [m/s, m/s, rad/s]	v_x, v_y, w_z
			ob = [],
			goal = [0.0, 0.0]			# [m, m] x, y
			):
		"""
		"""
		return [pose[0], pose[1], pose[2], u[0], u[1], u[2]], ob, goal


	def motion(self, x, u):
		"""
		motion model
		"""
		y = [0, 0, 0, 0, 0, 0]
		y[2] = x[2] + u[2] * self.dt
		y[0] = x[0] + (u[0] * math.cos(x[2]) * self.dt) - (u[1] * math.sin(x[2]) * self.dt)
		y[1] = x[1] + (u[0] * math.sin(x[2]) * self.dt) + (u[1] * math.cos(x[2]) * self.dt)
		y[3] = u[0]
		y[4] = u[1]
		y[5] = u[2]

		return y

	def calc_dynamic_window(self, x):
		"""
		calculation dynamic window based on current state x
		"""
		# Dynamic window from motion model
		Vd = [x[3] - self.max_accel_x * self.dt,
			  x[3] + self.max_accel_x * self.dt,
			  x[4] - self.max_accel_y * self.dt,
			  x[4] + self.max_accel_y * self.dt,
			  x[5] - self.max_delta_yaw_rate * self.dt,
			  x[5] + self.max_delta_yaw_rate * self.dt]

		dw = [
			max(self.Vs[0], Vd[0]), min(self.Vs[1], Vd[1]),
			max(self.Vs[2], Vd[2]), min(self.Vs[3], Vd[3]),
			max(self.Vs[4], Vd[4]), min(self.Vs[5], Vd[5])
			]

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
		best_u = [0.0, 0.0, 0.0]
		best_trajectory = [x]

		# evaluate all trajectory with sampled input in dynamic window
		print("dw: ", dw)
		for v_x in np.arange(dw[0], dw[1], self.v_x_resolution):
			for v_y in np.arange(dw[2], dw[3], self.v_y_resolution):
				# for w in np.arange(dw[4], dw[5], self.yaw_rate_resolution):

				trajectory = self.predict_trajectory(x, v_x, v_y, 0.0)

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
					best_u = [v_x, v_y, 0.0]
					best_trajectory = trajectory
					best_angle = angle
					if abs(best_u[0]) < self.robot_stuck_flag_cons \
							and abs(x[3]) < self.robot_stuck_flag_cons:
						# to ensure the robot do not get stuck in
						# best v=0 m/s (in front of an obstacle) and
						# best omega=0 rad/s (heading to the goal with
						# angle difference of 0)
						best_u[2] = -self.max_yaw_rate

		# print("Best_traj: ", best_trajectory)
		print("Min_cost: ", min_cost)
		print("Min_angle: ", best_angle)
		print("min_heading_cost: ", min_heading_cost)
		print("min_speed_cost: ", min_speed_cost)
		print("min_ob_cost: ", min_ob_cost)
		print("------------------------\n")

		return best_u, best_trajectory

	def predict_trajectory(self, x, v_x, v_y, w):
		"""
		predict trajectory with an input
		"""
		trajectory = [x]
		for i in range(int(self.predict_time/self.dt)):
			x = self.motion(x, [v_x, v_y, w])
			trajectory.append(x)

		return trajectory

	def calc_vel_cost(self, trajectory):
		"""
		"""
		return abs(self.max_speed_x - np.hypot(trajectory[-1][3],trajectory[-1][4]))
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
			if np.any(np.transpose(r) < self.collision_vector):
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

		# return angle

		# Using distance from goal: best performances
		dx = goal[0] - trajectory[-1][0]
		dy = goal[1] - trajectory[-1][1]
		distance = np.hypot(dx,dy)

		return distance


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