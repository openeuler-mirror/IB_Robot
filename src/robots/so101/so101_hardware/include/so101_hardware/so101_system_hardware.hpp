// Copyright 2026 IB_Robot Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
#define SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "ibrobot_msgs/msg/joint_current.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "so101/arm.hpp"

namespace so101_hardware
{

// Thin ros2_control adapter over the so101_sdk Arm. All protocol handling,
// calibration loading/validation, activation rollback, and unit conversion
// live in the SDK; this class only maps ros2_control interface buffers to
// the SDK's named-joint API and publishes joint currents.
//
// Relocated coverage (formerly in this header's detail namespace):
// - rollback_activation / disable_torque_on_abort -> feetech_sdk
//   (test_rollback.cpp)
// - SafeSMSSTS serial hardening -> feetech_sdk (test_serial_hardening.cpp)
// - perform_initial_sync_feedback -> so101_sdk Arm::activate
//   (test_arm.cpp: ActivationAbortLeavesSafeState)
class SO101SystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(SO101SystemHardware)

  hardware_interface::CallbackReturn
  on_init(const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface>
  export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface>
  export_command_interfaces() override;
  hardware_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn
  on_error(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::return_type perform_command_mode_switch(
    const std::vector<std::string> & start_interfaces,
    const std::vector<std::string> & stop_interfaces) override;
  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;
  hardware_interface::return_type
  write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  friend struct SO101HardwareTestAccess;
  void publish_currents(const rclcpp::Time & stamp);

  so101::ArmConfig arm_config_;
  // Declared before arm_ so the non-owning Arm attachment is destroyed first.
  std::unique_ptr<feetech::Bus> bus_;
  std::unique_ptr<so101::Arm> arm_;
  std::vector<std::string> joint_names_;
  std::map<std::string, double> command_map_;  // persistent: no per-cycle alloc
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_currents_;
  std::vector<double> hw_commands_;
  rclcpp::Node::SharedPtr current_node_;
  rclcpp::Publisher<ibrobot_msgs::msg::JointCurrent>::SharedPtr current_pub_;

  static constexpr double WRITE_RETRY_BUDGET_RATIO = 0.4;
  static constexpr auto FAILURE_TOLERANCE = std::chrono::milliseconds(200);
  std::chrono::steady_clock::duration write_retry_budget_{};
  std::optional<std::chrono::steady_clock::time_point> first_read_failure_;
  std::optional<std::chrono::steady_clock::time_point> first_write_failure_;
};

}  // namespace so101_hardware

#endif  // SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
