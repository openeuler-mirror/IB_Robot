#include "lekiwi_hardware/lekiwi_system_hardware.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "SMS_STS.h"
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>

namespace lekiwi_hardware
{

static constexpr double TICKS_PER_RAD = 4096.0 / (2.0 * M_PI);

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
    return hardware_interface::CallbackReturn::ERROR;

  const size_t n = info_.joints.size();
  if (n != NUM_JOINTS) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Expected %zu joints, got %zu", NUM_JOINTS, n);
    return hardware_interface::CallbackReturn::ERROR;
  }

  port_ = info_.hardware_parameters["port"];
  calib_file_ = info_.hardware_parameters["calib_file"];

  hw_positions_.resize(n, 0.0);
  hw_velocities_.resize(n, 0.0);
  hw_commands_.resize(n, 0.0);
  motor_ids_.resize(n);

  // Arm write buffers
  arm_target_positions_.resize(NUM_ARM_JOINTS, 0);
  arm_target_speeds_.resize(NUM_ARM_JOINTS, 0);
  arm_target_accs_.resize(NUM_ARM_JOINTS, 0);

  for (size_t i = 0; i < n; i++) {
    motor_ids_[i] = std::stoi(info_.joints[i].parameters.at("id"));
    if (i < NUM_ARM_JOINTS) {
      arm_motor_ids_.push_back(motor_ids_[i]);
    } else {
      base_motor_ids_.push_back(motor_ids_[i]);
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"),
    "Initialized: %zu arm joints + %zu base joints on port %s",
    arm_motor_ids_.size(), base_motor_ids_.size(), port_.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_configure(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Configuring...");

  std::ifstream f(calib_file_);
  if (!f.is_open()) {
    RCLCPP_WARN(rclcpp::get_logger("LeKiwiSystemHardware"),
      "Calibration file not found: %s", calib_file_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  auto calib = nlohmann::json::parse(f);

  // Build a lookup table: motor id -> calibration entry
  // The JSON keys can be motor IDs ("1","2") or names ("shoulder_pan","arm_shoulder_pan"),
  // so we match by the "id" field inside each entry.
  std::map<u8, nlohmann::json> calib_by_id;
  for (auto & [key, entry] : calib.items()) {
    if (entry.contains("id")) {
      u8 mid = entry["id"].get<u8>();
      calib_by_id[mid] = entry;
    }
  }

  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 id = motor_ids_[i];
    // Base motors (wheel mode) don't need calibration data
    if (i >= NUM_ARM_JOINTS) {
      homing_offsets_[id] = 0;
      range_mins_[id] = 0;
      range_maxes_[id] = 4095;
      continue;
    }
    if (calib_by_id.find(id) == calib_by_id.end()) {
      RCLCPP_ERROR(rclcpp::get_logger("LeKiwiSystemHardware"),
        "Calibration entry not found for arm motor %d", id);
      return hardware_interface::CallbackReturn::ERROR;
    }
    auto & entry = calib_by_id[id];
    homing_offsets_[id] = entry["homing_offset"];
    range_mins_[id] = entry["range_min"];
    range_maxes_[id] = entry["range_max"];
  }

  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Configured!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> LeKiwiSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> LeKiwiSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    // Arm joints (0-5): position command; Base joints (6-8): velocity command
    if (i < NUM_ARM_JOINTS) {
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
    } else {
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_[i]);
    }
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Activating...");

  if (!sms_sts_.begin(1000000, port_.c_str())) {
    RCLCPP_ERROR(rclcpp::get_logger("LeKiwiSystemHardware"),
      "Failed to connect to motors on port %s", port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Give motors time to initialize after serial connection
  usleep(500000); // 500ms delay

  // Ping each motor
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 id = motor_ids_[i];
    int retry = 3;
    bool found = false;
    while (retry--) {
      if (sms_sts_.Ping(id) != -1) {
        found = true;
        break;
      }
      usleep(10000);
    }
    if (!found) {
      RCLCPP_ERROR(rclcpp::get_logger("LeKiwiSystemHardware"),
        "Motor ID %d is NOT responding!", id);
      return hardware_interface::CallbackReturn::FAILURE;
    }
    RCLCPP_DEBUG(rclcpp::get_logger("LeKiwiSystemHardware"), "Motor ID %d found.", id);
  }

  // Configure arm motors (IDs 1-6): position mode
  for (size_t i = 0; i < arm_motor_ids_.size(); i++) {
    u8 id = arm_motor_ids_[i];
    sms_sts_.EnableTorque(id, 0);
    usleep(2000);
    sms_sts_.unLockEprom(id);
    usleep(2000);

    int offset = homing_offsets_[id];
    u16 encoded_offset = (offset < 0)
      ? (static_cast<u16>(std::abs(offset)) | (1 << 11))
      : static_cast<u16>(offset);
    sms_sts_.writeWord(id, 31, encoded_offset);
    sms_sts_.writeWord(id, 9, range_mins_[id]);
    sms_sts_.writeWord(id, 11, range_maxes_[id]);
    sms_sts_.writeByte(id, 7, 0);
    sms_sts_.writeByte(id, 21, 16);
    sms_sts_.writeByte(id, 22, 32);
    sms_sts_.writeByte(id, 23, 0);

    sms_sts_.LockEprom(id);
    usleep(2000);
    sms_sts_.EnableTorque(id, 1);
    usleep(2000);
  }

  // Configure base motors (IDs 7-9): wheel (velocity) mode
  for (size_t i = 0; i < base_motor_ids_.size(); i++) {
    u8 id = base_motor_ids_[i];
    sms_sts_.EnableTorque(id, 0);
    usleep(2000);
    sms_sts_.unLockEprom(id);
    usleep(2000);

    // Set to wheel mode (continuous rotation)
    sms_sts_.WheelMode(id);
    usleep(2000);

    sms_sts_.LockEprom(id);
    usleep(2000);
    sms_sts_.EnableTorque(id, 1);
    usleep(2000);
  }

  // Initialize sync read for all 9 motors (position: 2 bytes at register 56)
  sms_sts_.syncReadBegin(motor_ids_.size(), 2, 10);

  // Initial read for arm positions
  if (sms_sts_.syncReadPacketTx(motor_ids_.data(), motor_ids_.size(), SMS_STS_PRESENT_POSITION_L, 2) > 0) {
    for (size_t i = 0; i < motor_ids_.size(); i++) {
      u8 data[2];
      if (sms_sts_.syncReadPacketRx(motor_ids_[i], data) == 2) {
        s16 pos = (data[1] << 8) | data[0];
        if (pos & (1 << 15)) pos = -(pos & 0x7FFF);  // sign bit
        double rad = (static_cast<double>(pos) - 2048.0) / TICKS_PER_RAD;
        hw_positions_[i] = rad;
        if (i < NUM_ARM_JOINTS) {
          hw_commands_[i] = rad;  // hold current position
        }
      }
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"),
    "Activated! %zu arm + %zu base motors running.",
    arm_motor_ids_.size(), base_motor_ids_.size());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Deactivating...");

  // Stop base motors first
  if (!base_motor_ids_.empty()) {
    s16 zero_speeds[3] = {0, 0, 0};
    u8 zero_accs[3] = {0, 0, 0};
    sms_sts_.SyncWriteSpe(base_motor_ids_.data(), base_motor_ids_.size(), zero_speeds, zero_accs);
  }

  // Disable torque on all motors
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    sms_sts_.EnableTorque(motor_ids_[i], 0);
  }
  usleep(100000);
  sms_sts_.syncReadEnd();
  sms_sts_.end();

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type LeKiwiSystemHardware::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  // Read positions from all motors
  int read_len = sms_sts_.syncReadPacketTx(
    motor_ids_.data(), motor_ids_.size(), SMS_STS_PRESENT_POSITION_L, 2);
  if (read_len <= 0) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("LeKiwiSystemHardware"), steady_clock, 500,
      "SyncRead PacketTx FAILED");
    return hardware_interface::return_type::OK;
  }

  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 data[2];
    if (sms_sts_.syncReadPacketRx(motor_ids_[i], data) == 2) {
      s16 pos = (data[1] << 8) | data[0];
      if (pos & (1 << 15)) pos = -(pos & 0x7FFF);

      if (i < NUM_ARM_JOINTS) {
        // Arm: convert ticks to radians
        hw_positions_[i] = (static_cast<double>(pos) - 2048.0) / TICKS_PER_RAD;
      } else {
        // Base: store raw position tick (accumulated rotation in wheel mode)
        hw_positions_[i] = static_cast<double>(pos);
      }
    }
  }

  // Also read speeds for arm motors
  read_len = sms_sts_.syncReadPacketTx(
    motor_ids_.data(), motor_ids_.size(), SMS_STS_PRESENT_SPEED_L, 2);
  if (read_len > 0) {
    for (size_t i = 0; i < motor_ids_.size(); i++) {
      u8 data[2];
      if (sms_sts_.syncReadPacketRx(motor_ids_[i], data) == 2) {
        s16 speed = (data[1] << 8) | data[0];
        if (speed & (1 << 15)) speed = -(speed & 0x7FFF);

        if (i < NUM_ARM_JOINTS) {
          // Arm velocity in raw steps/s → convert to rad/s
          hw_velocities_[i] = static_cast<double>(speed) / TICKS_PER_RAD;
        } else {
          // Base velocity: convert raw steps/s → rad/s for ros2_control state interface
          hw_velocities_[i] = static_cast<double>(speed) / TICKS_PER_RAD;
        }
      }
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type LeKiwiSystemHardware::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  // ---- Arm motors: position control (SyncWritePosEx) ----
  for (size_t i = 0; i < NUM_ARM_JOINTS; i++) {
    double target_raw = hw_commands_[i] * TICKS_PER_RAD + 2048.0;
    // Safety clamp
    if (target_raw < 0) target_raw = 0;
    if (target_raw > 4095) target_raw = 4095;
    arm_target_positions_[i] = static_cast<s16>(target_raw);
    arm_target_speeds_[i] = 2400;
    arm_target_accs_[i] = 50;
  }

  if (!arm_motor_ids_.empty()) {
    sms_sts_.SyncWritePosEx(
      arm_motor_ids_.data(), arm_motor_ids_.size(),
      arm_target_positions_.data(), arm_target_speeds_.data(), arm_target_accs_.data());
  }

  // ---- Base motors: velocity control (SyncWriteSpe) ----
  if (!base_motor_ids_.empty()) {
    s16 base_speeds[NUM_BASE_JOINTS];
    u8 base_accs[NUM_BASE_JOINTS];
    for (size_t i = 0; i < NUM_BASE_JOINTS; i++) {
      // hw_commands_[NUM_ARM_JOINTS + i] is velocity in rad/s (ros2_control convention).
      // Convert to raw steps/s for the STS3215 speed register.
      double raw = hw_commands_[NUM_ARM_JOINTS + i] * TICKS_PER_RAD;
      // Clamp to safe s16 range
      if (raw > 32767) raw = 32767;
      if (raw < -32768) raw = -32768;
      base_speeds[i] = static_cast<s16>(raw);
      base_accs[i] = 50;
    }
    sms_sts_.SyncWriteSpe(
      base_motor_ids_.data(), base_motor_ids_.size(), base_speeds, base_accs);
  }

  return hardware_interface::return_type::OK;
}

}  // namespace lekiwi_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  lekiwi_hardware::LeKiwiSystemHardware, hardware_interface::SystemInterface)
