#!/usr/bin/env python3

import time
import sys
import rclpy
from rclpy.node import Node
import rclpy.qos as qos

from rclpy.parameter import Parameter
# from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterValue

import numpy as np


class DWBparamsClient(Node):
    def __init__(self):
        super().__init__('dwb_client')
        #rclpy.logging.set_logger_level('dwb_client', 10)

        self.get_cli_controller = self.create_client(GetParameters, '/controller_server/get_parameters')
        while not self.get_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_req_controller = GetParameters.Request()

        self.get_cli_costmap = self.create_client(GetParameters, '/local_costmap/local_costmap/get_parameters')
        while not self.get_cli_costmap.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_req_costmap = GetParameters.Request()

        self.set_cli_controller = self.create_client(SetParameters, '/controller_server/set_parameters')
        while not self.set_cli_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_controller = SetParameters.Request()

        self.set_cli_costmap = self.create_client(SetParameters, '/local_costmap/local_costmap/set_parameters')
        while not self.set_cli_costmap.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_req_costmap = SetParameters.Request()

    def send_get_request_controller(self):
        self.get_req_controller.names = ['FollowPath.max_vel_x','FollowPath.max_vel_theta','FollowPath.vx_samples','FollowPath.vtheta_samples','FollowPath.BaseObstacle.scale','FollowPath.PathDist.scale','FollowPath.GoalDist.scale']
        future = self.get_cli_controller.call_async(self.get_req_controller)
        return future

    def send_get_request_costmap(self):
        self.get_req_costmap.names = ['inflation_layer.inflation_radius']
        future = self.get_cli_costmap.call_async(self.get_req_costmap)
        return future
        

    def send_set_request_controller(self, param_values):
        self.set_req_controller.parameters = [Parameter(name='FollowPath.max_vel_x', value=param_values[0]).to_parameter_msg(),
                                              Parameter(name='FollowPath.max_vel_theta', value=param_values[1]).to_parameter_msg(),
                                              Parameter(name='FollowPath.vx_samples', value=param_values[2]).to_parameter_msg(),
                                              Parameter(name='FollowPath.vtheta_samples', value=param_values[3]).to_parameter_msg(),
                                              Parameter(name='FollowPath.BaseObstacle.scale', value=param_values[4]).to_parameter_msg(),
                                              Parameter(name='FollowPath.PathDist.scale', value=param_values[5]).to_parameter_msg(),
                                              Parameter(name='FollowPath.GoalDist.scale', value=param_values[6]).to_parameter_msg()]
        future = self.set_cli_controller.call_async(self.set_req_controller)
        return future

    def send_set_request_costmap(self, param_value):
        self.set_req_costmap.parameters = [Parameter(name='inflation_layer.inflation_radius', value=param_value).to_parameter_msg()]
        future = self.set_cli_costmap.call_async(self.set_req_costmap)
        return future

    def send_params_action(self, dwb_params):
        controller_params = dwb_params[:-1]
        #self.get_logger().info('setting controller_params to: '+str(controller_params))
        costmap_param = dwb_params[-1]
        #self.get_logger().info('setting inflation_radius to: '+str(costmap_param))

        self.set_req_controller = SetParameters.Request()
        future = self.send_set_request_controller(controller_params)
        rclpy.spin_until_future_complete(self, future)

        try:
            get_response = future.result()
            self.get_logger().debug(
                'Result %s' %
                (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug(
                'Service call failed %r' % (e,))

        # while rclpy.ok():
        #     rclpy.spin_once(self)
        #     if future.done():
        #         try:
        #             get_response = future.result()
        #             self.get_logger().info(
        #                 'Result %s' %
        #                 (get_response.results[0].successful))
        #         except Exception as e:
        #             self.get_logger().info(
        #                 'Service call failed %r' % (e,))
        #             break

        self.set_req_costmap = SetParameters.Request()

        future = self.send_set_request_costmap(costmap_param)
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().debug(
                'Result %s' %
                (get_response.results[0].successful))
        except Exception as e:
            self.get_logger().debug(
                'Service call failed %r' % (e,))

        # while rclpy.ok():
        #     rclpy.spin_once(self)
        #     if future.done():
        #         try:
        #             get_response = future.result()
        #             self.get_logger().info(
        #                 'Result %s' %
        #                 (get_response.results[0].successful))
        #         except Exception as e:
        #             self.get_logger().info(
        #                 'Service call failed %r' % (e,))
        #         break

    def get_dwb_params(self,):

        future = self.send_get_request_controller()
        rclpy.spin_until_future_complete(self, future)
        try:
            get_response = future.result()
            self.get_logger().info(
                    'Result %s %s %s %s %s %s %s' %(
                    get_response.values[0].double_value,
                    get_response.values[1].double_value, 
                    get_response.values[2].integer_value, 
                    get_response.values[3].integer_value,
                    get_response.values[4].double_value,
                    get_response.values[5].double_value,
                    get_response.values[6].double_value
                    ))

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        future = self.send_get_request_costmap()

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    get_response = future.result()
                    self.get_logger().info(
                        'Result %s ' %
                        (get_response.values[0].double_value))
                except Exception as e:
                    self.get_logger().info(
                        'Service call failed %r' % (e,))
                break


def main(args=None):
    rclpy.init(args=args)

    dwb_client = DWBparamsClient()

    for i in range(5):
        values = np.random.random()
        print(values)
        future = dwb_client.send_set_request_controller(values)
        
        while rclpy.ok():
            rclpy.spin_once(dwb_client)
            if future.done():
                try:
                    get_response = future.result()
                    dwb_client.get_logger().info(
                        'Result %s' %
                        (get_response.results[0].successful))
                except Exception as e:
                    dwb_client.get_logger().info(
                        'Service call failed %r' % (e,))
                break

        future = dwb_client.send_get_request_controller()
        while rclpy.ok():
            rclpy.spin_once(dwb_client)
            if future.done():
                try:
                    get_response = future.result()
                    # dwb_client.get_logger().info(
                    #     'Result %s %s %s' %
                    #     (get_response.values[0].double_value, get_response.values[1].integer_value, get_response.values[2].double_value))
                    dwb_client.get_logger().info(
                        'Result %s ' %
                        (get_response.values[1].double_value))
                except Exception as e:
                    dwb_client.get_logger().info(
                        'Service call failed %r' % (e,))
                break

    dwb_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()