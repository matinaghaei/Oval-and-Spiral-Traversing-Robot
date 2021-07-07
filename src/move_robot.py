#! /usr/bin/env python3

# Import the Python library for ROS
import rospy
import time
import numpy as np
import matplotlib.pyplot as plt

# Import the Odometry message
from nav_msgs.msg import Odometry

# Import the Twist message
from geometry_msgs.msg import Twist

# TF allows to perform transformations between different coordinate frames
import tf

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState

path_shape = 'oval'
oval_x = 1
oval_y = 3
spiral_growth_factor = 0.1
robot_x = 1
robot_y = 2

k_p = 0.3
k_i = 0.3
k_theta = 3
window_size = 100

class MoveRobot():

	def __init__(self):
		# Initiate a named node
		rospy.init_node('MoveRobot', anonymous=False)
		
		# tell user how to stop TurtleBot
		rospy.loginfo("CTRL + C to stop the turtlebot")
		
		# What function to call when ctrl + c is issued
		rospy.on_shutdown(self.shutdown)
		
		rospy.wait_for_service("gazebo/get_model_state")
		self.get_ground_truth = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
		
		# Create a Publisher object, will publish on cmd_vel_mux/input/teleop topic
		# to which the robot (real or simulated) is a subscriber
		self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
		
		# Creates a var of msg type Twist for velocity
		self.vel = Twist()
		
		# Set a publish velocity rate of in Hz
		self.rate = rospy.Rate(5)
		
		self.points = []
		if path_shape == 'oval':
			self.ds = 0.1
			angles = np.arange(0, 2*np.pi, np.pi/25)
			for theta in angles:
				point = [oval_x*np.cos(theta), oval_y*np.sin(theta)]
				self.points.append(point)
				plt.scatter(*point, s=40, c='orange')
		elif path_shape == 'spiral':
			self.ds = 0.25
			angles = np.arange(0, 6*np.pi, np.pi/25)
			for theta in angles:
				point = [(spiral_growth_factor*theta+0.5)*np.cos(theta), (spiral_growth_factor*theta+0.5)*np.sin(theta)]
				self.points.append(point)
				plt.scatter(*point, s=40, c='orange')
		self.next_point = 0

		self.odom_sub = rospy.Subscriber('/odom', Odometry, self.callback_odometry)

		self.path_x = []
		self.path_y = []
		
		self.pos_diffs = []
		
		self.is_finished = False
		
		self.set_position(robot_x, robot_y)
		self.prevp = (robot_x, robot_y)

	def measure_error(self):
		if self.next_point == len(self.points):
			return
		position = self.get_ground_truth("turtlebot3_burger", "world").pose.position
		current = (position.x, position.y)
		prevp = self.prevp
		nextp = self.points[self.next_point]
		dist = np.sqrt((nextp[0]-prevp[0])**2+(nextp[1]-prevp[1])**2)
		prevp_error = np.sqrt((current[0]-prevp[0])**2+(current[1]-prevp[1])**2)
		nextp_error = np.sqrt((current[0]-nextp[0])**2+(current[1]-nextp[1])**2)
		if dist**2+prevp_error**2<=nextp_error**2:
			return prevp_error
		if dist**2+nextp_error**2<=prevp_error**2:
			return nextp_error
		return abs((nextp[0]-prevp[0])*(prevp[1]-current[1])-(prevp[0]-current[0])*(nextp[1]-prevp[1]))/dist

	def set_position(self, x, y):
		state_msg = ModelState()
		state_msg.model_name = 'turtlebot3_burger'
		state_msg.pose.position.x = x
		state_msg.pose.position.y = y
		state_msg.pose.position.z = 0
		state_msg.pose.orientation.x = 0
		state_msg.pose.orientation.y = 0
		state_msg.pose.orientation.z = 0
		state_msg.pose.orientation.w = 0

		rospy.wait_for_service('/gazebo/set_model_state')
		try:
			set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			resp = set_state( state_msg )
		except rospy.ServiceException as e:
			print(f"Service call failed: {e}")

	def callback_odometry(self, msg):
		if self.next_point == len(self.points):
			return
		current = (msg.pose.pose.position.x, msg.pose.pose.position.y)
		self.path_x.append(current[0])
		self.path_y.append(current[1])
		target = self.points[self.next_point]
		pos_diff = np.sqrt((current[0]-target[0])**2 + (current[1]-target[1])**2)
		while pos_diff < self.ds:
			print(f"************************* Reached Point #{self.next_point} *************************")
			self.prevp = current
			self.next_point += 1
			if self.next_point == len(self.points):
				self.shutdown()
				return
			target = self.points[self.next_point]
			pos_diff = np.sqrt((current[0]-target[0])**2 + (current[1]-target[1])**2) - self.ds
		self.pos_diffs.append(pos_diff)
		window = self.pos_diffs[-window_size:]
		pos_diff_integral = sum(window) / len(window)
		self.vel.linear.x = k_p * pos_diff + k_i * pos_diff_integral
		
		target_yaw = np.arctan2(target[1] - current[1], target[0] - current[0])
		quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
			msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		(roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
		yaw_diff = target_yaw - yaw
		while not -np.pi <= yaw_diff <= np.pi:
			if yaw_diff > np.pi:
				yaw_diff -= 2 * np.pi
			if yaw_diff < -np.pi:
				yaw_diff += 2 * np.pi
		self.vel.angular.z = k_theta * yaw_diff
		
	def send_velocity_cmd(self):
		self.vel_pub.publish(self.vel)
	
	def shutdown(self):
		print("Shutdown!")
		# stop TurtleBot
		rospy.loginfo("Stop TurtleBot")
		
		self.vel.linear.x = 0.0
		self.vel.angular.z = 0.0
		
		self.vel_pub.publish(self.vel)
		
		# makes sure robot receives the stop command prior to shutting down
		rospy.sleep(1)
		
		self.is_finished = True
		
if __name__ == '__main__':
	
	path_shape = input("Enter path shape: ")
	if path_shape == 'oval':
		oval_x = float(input("Enter the x axis of oval: "))
		oval_y = float(input("Enter the y axis of oval: "))
	elif path_shape == 'spiral':
		spiral_growth_factor = float(input("Enter the growth factor of spiral: "))
	robot_x = float(input("Enter robot position (x axis): "))
	robot_y = float(input("Enter robot position (y axis): "))

	errors = []

	try:
		controller = MoveRobot()
		
		# keeping doing until ctrl+c
		while not rospy.is_shutdown() and not controller.is_finished:
			
			error = controller.measure_error()
			if error:
				errors.append(error)
				print("Current Error:", error)
			
			# send velocity commands to the robots
			controller.send_velocity_cmd()
			
			# wait for the selected mseconds and publish velocity again
			controller.rate.sleep()

		print("Average Error:", sum(errors) / len(errors))
		plt.scatter(controller.path_x, controller.path_y, s=0.5, c='r')
		plt.show()
		
	except:
		rospy.loginfo("move_robot node terminated")

