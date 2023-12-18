#include "applr_social_nav/pause_navigation.hpp"
#include <chrono>
#include <memory>
#include <string>

namespace applr_social_nav
{

PauseNavigation::PauseNavigation(
  const std::string & condition_name,
  const BT::NodeConfiguration & conf)
: BT::ConditionNode(condition_name, conf)
{
  node_ = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
  server_ = node_->create_service<std_srvs::srv::SetBool>(
    "pause_navigation",
    std::bind(&PauseNavigation::service_callback, this, std::placeholders::_1, std::placeholders::_2));

}

BT::NodeStatus PauseNavigation::tick()
{
  if (is_paused_) {
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::FAILURE;
}

void PauseNavigation::service_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  is_paused_ = request->data;
  if(is_paused_)
  {
    RCLCPP_INFO(node_->get_logger(), "Received request to pause navigation");
  }
  else
  {
    RCLCPP_INFO(node_->get_logger(), "Received request to resume navigation");
  }
  response->success = true;
}

}  // namespace applr_social_nav

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<applr_social_nav::PauseNavigation>("PauseNavigation");
}