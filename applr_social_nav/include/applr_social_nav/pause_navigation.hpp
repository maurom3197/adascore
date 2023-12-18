#ifndef APPLR_SOCIAL_NAV__PAUSE_NAVIGATION_HPP_
#define APPLR_SOCIAL_NAV__PAUSE_NAVIGATION_HPP_

#include <string>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "behaviortree_cpp_v3/condition_node.h"
#include "std_srvs/srv/set_bool.hpp"

namespace applr_social_nav
{

/**
 * @brief A BT::ConditionNode that returns SUCCESS when the navigation stack is paused
   and FAILURE otherwise
 */
class PauseNavigation : public BT::ConditionNode
{
public:
  /**
   * @brief A constructor for applr_social_nav::PauseNavigation
   * @param condition_name Name for the XML tag for this node
   * @param conf BT node configuration
   */
  PauseNavigation(
    const std::string & condition_name,
    const BT::NodeConfiguration & conf);

    PauseNavigation() = delete;

  /**
   * @brief The main override required by a BT action
   * @return BT::NodeStatus Status of tick execution
   */
  BT::NodeStatus tick() override;

  /**
   * @brief Creates list of BT ports
   * @return BT::PortsList Containing node-specific ports
   */
  static BT::PortsList providedPorts()
  {
    return {};
  }

private:
  void service_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr server_;

  bool is_paused_;
};

}  // namespace applr_social_nav

#endif  // APPLR_SOCIAL_NAV__PAUSE_NAVIGATION_HPP_