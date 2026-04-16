#ifndef LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_
#define LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_

#include <map>
#include <string>
#include <vector>
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "SMS_STS.h"

namespace lekiwi_hardware
{

// Number of arm joints (position-controlled)
static constexpr size_t NUM_ARM_JOINTS = 6;
// Number of base joints (velocity-controlled omniwheels)
static constexpr size_t NUM_BASE_JOINTS = 3;
// Total joints
static constexpr size_t NUM_JOINTS = NUM_ARM_JOINTS + NUM_BASE_JOINTS;

class LeKiwiSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(LeKiwiSystemHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  SMS_STS sms_sts_;
  std::string port_;
  std::string calib_file_;

  // State buffers for all 9 joints
  std::vector<double> hw_positions_;   // arm positions (rad) + base velocities (raw)
  std::vector<double> hw_velocities_;  // arm velocities (rad/s) + base velocities (raw)

  // Command buffers for all 9 joints
  std::vector<double> hw_commands_;    // arm: position (rad), base: velocity (raw)

  // Motor IDs
  std::vector<u8> motor_ids_;
  // Separate ID arrays for sync operations
  std::vector<u8> arm_motor_ids_;
  std::vector<u8> base_motor_ids_;

  // Arm write buffers
  std::vector<s16> arm_target_positions_;
  std::vector<u16> arm_target_speeds_;
  std::vector<u8> arm_target_accs_;

  // Calibration data
  std::map<u8, int> homing_offsets_;
  std::map<u8, int> range_mins_;
  std::map<u8, int> range_maxes_;
};

}  // namespace lekiwi_hardware

#endif  // LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_
