#!/usr/bin/env python3

import os
# Python libraries
import numpy as np
import math
import json
import yaml
import logging
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from people_msgs.msg import People
from pic4rl.utils.env_utils import normalize, normalize_angle
from adascore.utils.nav_utils import filter_people, process_people

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


class Agent():
    def __init__(
        self,
        # desiredVelocity,
        # radius,
        # cyclicGoals,
        # teleoperated, 
        # antimove, 
        # linearVelocity,
        # angularVelocity, 
        # groupId
        ):

        # self.agent_params = dict([
        #     ('desiredVelocity': 0.6), 
        #     ('radius': 0.35), 
        #     ('cyclicGoals': False),
        #     ('teleoperated': False), 
        #     ('antimove': False), 
        #     ('linearVelocity': 0),
        #     ('angularVelocity': 0), 
        #     ('groupId':-1),
        #     ])

        self.agent_name = 'agent0'
        self.id = 0
        self.max_vel = 1.5
        self.desiredVelocity = 0.6
        self.radius = 0.4
        self.goal_radius = 0.3
        self.cyclicGoals = True
        self.teleoperated = False
        self.antimove = False
        self.linearVelocity = 0.
        self.angularVelocity = 0.
        self.group_id = -1

        self.forceFactorDesired = 2.0
        self.forceFactorObstacle = 10
        self.forceSigmaObstacle = 0.2
        self.forceFactorSocial = 2.1
        self.forceFactorGroupGaze = 3.0
        self.forceFactorGroupCoherence = 2.0
        self.forceFactorGroupRepulsion = 1.0
        self.lambda_ = 2.0
        self.gamma = 0.35
        self.n = 2.0
        self.nPrime = 3.0
        self.relaxationTime = 0.5

        self.position = [0., 0.] # [x, y]
        self.yaw = 0.0
        self.velocity = [0., 0.] # [v_x, v_y]
        self.ang_velocity = 0.0
        self.goals = [[0., 0.],]
        self.current_goal = [0.,0.]

        self.socialForce = np.zeros(2, dtype=np.float32)
        self.obstacleForce = np.zeros(2, dtype=np.float32)
        self.groupForce = np.zeros(2, dtype=np.float32)

    def update_state(self, position, velocity):
        self.position = position[:-1]
        self.velocity = velocity[:-1]
        self.yaw = position[-1]
        self.ang_velocity = velocity[-1]
        #self.current_goal = goal


class SocialForceModel():
    def __init__(
        self,
        node,
        agents_config
        ):

        self.agents = []
        self.node = node
        self.node.get_logger().info('Instanciating Social Force Model')
        #agents_config = 'social_indoor_agents.yaml'
        self.config_params = self.get_param(agents_config)

        self.num_agents = len(self.config_params["agents"]) # something similar
        self.instanciate_agents()

        self.people_msg = People()

        self.node.get_logger().info('People topic subscription')
        self.people_sub = self.node.create_subscription(
                People,
                '/people', 
                self.people_callback,
                1)

    def get_param(self, agents_config):
        configFilepath = os.path.join(
            get_package_share_directory("adascore"), 'config', "agents_envs", 
            agents_config)
                            
        # Load the topic parameters
        with open(configFilepath, 'r') as file:
            configParams = yaml.safe_load(file)['hunav_loader']['ros__parameters']

        return configParams

    def people_callback(self, msg):
        """
        """
        self.people_msg = msg

    def instanciate_agents(self,):
        robot = Agent()
        robot.agent_name = "robot"
        robot.max_vel = 0.8
        self.agents.append(robot)

        for i in range(self.num_agents):
            agent = Agent()
            agent.agent_name = "agent"+str(i+1)
            agent.id = self.config_params[agent.agent_name]["id"]
            agent.group_id = self.config_params[agent.agent_name]["group_id"]
            agent.max_vel = self.config_params[agent.agent_name]["max_vel"]
            agent.radius = self.config_params[agent.agent_name]["radius"]
            agent.goal_radius = self.config_params[agent.agent_name]["goal_radius"]
            agent.cyclic_goals = self.config_params[agent.agent_name]["cyclic_goals"]
            goals = self.config_params[agent.agent_name]["goals"]

            goal_list = []
            for g in goals:
                x = self.config_params[agent.agent_name][g]["x"]
                y = self.config_params[agent.agent_name][g]["y"]
                goal_list.append([x,y])
            agent.goals = goal_list

            self.agents.append(agent)

        print("Agents in sfm model: ", len(self.agents))

    def get_people(self, robot_pose, robot_velocity):
        """
        """
        people_state_ = []
        people_info_ = []
        distances = []
        msg = self.people_msg

        self.agents[0].update_state(robot_pose, robot_velocity)

        for i in range(len(self.agents[1:])):
            # get person pose
            x = msg.people[i].position.x
            y = msg.people[i].position.y
            yaw = msg.people[i].position.z
            person_pose = [x,y,yaw]

            # get person velocity
            vel_x = msg.people[i].velocity.x
            vel_y = msg.people[i].velocity.y
            vel_w = msg.people[i].velocity.z
            vel_module = math.sqrt((vel_x)**2+(vel_y)**2)

            # update agent state
            self.agents[i+1].update_state(person_pose, [vel_x, vel_y, vel_w])

            person_dist, person_angle = process_people(person_pose, robot_pose)

            if person_dist < self.node.max_person_dist_allowed:
                distances.append(person_dist)
                # people info is used to compute social forces
                people_info_.append([x,y,yaw,vel_x,vel_y,vel_w])
                # make variable relative to robot frame
                #x = x - robot_pose[0]
                #y = y - robot_pose[1]
                yaw = yaw - robot_pose[2]

                # Person state s_p as [x,y,yaw,vel_x,vel_y,vel_w] or [dist, angle, v_module, yaw]
                people_state_.append([person_dist, person_angle, vel_module, yaw])

        people_state, people_info_, min_people_distance = filter_people(self.node.k_people, distances, people_state_, people_info_)

        return people_state, people_info_, min_people_distance

    def computeDesiredForce(self, agent, goal_pose):

        #if (!agent.goals.empty() and (agent.goals.front().center - agent.position).norm() >
        #      agent.goals.front().radius) 

        diff = np.array(goal_pose) - np.array(agent.position)
        diff_norm = np.linalg.norm(diff)
        desiredDirection = diff / diff_norm

        agent.desiredForce = agent.forceFactorDesired*(desiredDirection * agent.desiredVelocity - agent.velocity) / agent.relaxationTime
        agent.antimove = False
        #else:
        #    agent.forces.desiredForce = -agent.velocity / agent.relaxationTime
        #    agent.antimove = True

        return desiredDirection

    def computeSocialForce_on_agent(self, index):
        ""
        ""
        #agent = self.agents[index]
        # forces are expressed as 2D vectors F=[fx,fy]
        social_force = np.zeros(2, dtype=np.float32)

        for i in range(len(self.agents)):
            if (i == index):
              continue

            diff = np.array(self.agents[i].position) - np.array(self.agents[index].position)
            diffDirection, diff_norm = normalize(diff)
            velDiff = np.array(self.agents[index].velocity) - np.array(self.agents[i].velocity)
            interactionVector = self.agents[index].lambda_ * velDiff + diffDirection
            interactionDirection, interactionLength = normalize(interactionVector)

            theta = math.atan2(\
                diffDirection[1], diffDirection[0])- math.atan2(\
                interactionDirection[1], interactionDirection[0])

            theta = normalize_angle(theta)
            B = self.agents[index].gamma * interactionLength

            # Compute force
            forceVelocityAmount = \
                -math.exp(-diff_norm / B - (self.agents[index].nPrime * B * theta)**2)
            
            forceAngleAmount = \
                -np.sign(theta) * math.exp(-diff_norm / B - (self.agents[index].n * B * theta)**2)

            forceVelocity = forceVelocityAmount * interactionDirection
            interactionDirection_leftNormal = np.array([-interactionDirection[1], interactionDirection[0]])
            forceAngle = forceAngleAmount * interactionDirection_leftNormal

            social_force = social_force \
                + self.agents[index].forceFactorSocial * (forceVelocity + forceAngle)

        self.agents[index].socialForce = social_force
        return social_force
        

    def computeSocialForce_by_robot(self, index):
        ""
        ""
        agent = self.agents[index]
        robot = self.agents[0]

        # forces are expressed as 2D vectors F=[fx,fy]
        social_force = np.zeros(2, dtype=np.float32)

        diff = np.array(robot.position) - np.array(agent.position)
        diffDirection, diff_norm = normalize(diff)
        velDiff = np.array(agent.velocity) - np.array(robot.velocity)
        interactionVector = agent.lambda_ * velDiff + diffDirection
        interactionDirection, interactionLength = normalize(interactionVector)

        theta = math.atan2(diffDirection[1], diffDirection[0]) \
              - math.atan2(interactionDirection[1], interactionDirection[0])

        theta = normalize_angle(theta)

        B = agent.gamma * interactionLength

        # Compute force
        forceVelocityAmount = \
            -math.exp(-diff_norm / B - (agent.nPrime * B * theta)**2)
        
        forceAngleAmount = \
            -np.sign(theta) * math.exp(-diff_norm / B - (agent.n * B * theta)**2)

        forceVelocity = forceVelocityAmount * interactionDirection
        interactionDirection_leftNormal = np.array([-interactionDirection[1], interactionDirection[0]])
        forceAngle = forceAngleAmount * interactionDirection_leftNormal

        social_force = agent.forceFactorSocial * (forceVelocity + forceAngle)

        return social_force


    def computeObstacleForce(self, agent, map):
        ""
        ""
        pass


    def computeGroupForce(self, agent, map):
        ""
        ""
        pass

    def computeSocialWork(self,):

        # social work on the robot
        sfr = self.computeSocialForce_on_agent(0)
        wr = np.linalg.norm(sfr)
            # +np.linalg.norm(self.agents[0].obstacleForce)
        #self.node.get_logger().debug("social work on the robot: "+str(wr))

        # Compute the social work provoked by the robot in the other agents
        wp = 0.
        for i in range(len(self.agents[1:])):
            i +=1
            social_force = self.computeSocialForce_by_robot(i)
            wp += np.linalg.norm(social_force)
            #self.node.get_logger().debug('social Force by the robot on ONE agent: '+str(np.linalg.norm(social_force)))
        #self.node.get_logger().debug('social Work by the robot on ALL agents: '+str(wp))

        return wr, wp