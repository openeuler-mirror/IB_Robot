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

#include "so101_hardware/so101_system_hardware.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include "feetech/conversion.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace so101_hardware
{


namespace
{
bool is_truthy(const std::string & value)
{
  return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

// One deadline covers every retry and every transport operation within it.
// Always release it before lifecycle safety operations can use the bus.
class ScopedIoDeadline
{
public:
  ScopedIoDeadline(feetech::Bus & bus, std::chrono::steady_clock::time_point deadline)
  : bus_(bus) {bus_.set_io_deadline(deadline);}
  ~ScopedIoDeadline() {bus_.clear_io_deadline();}
  ScopedIoDeadline(const ScopedIoDeadline &) = delete;
  ScopedIoDeadline & operator=(const ScopedIoDeadline &) = delete;

private:
  feetech::Bus & bus_;
};
}  // namespace

hardware_interface::CallbackReturn
SO101SystemHardware::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Build the SDK configuration from the URDF hardware parameters.
  so101::ArmConfig config;
  joint_names_.clear();
  command_map_.clear();
  gripper_opening_directions_.clear();
  try {
    const auto rate_param = info_.hardware_parameters.find("update_rate");
    size_t parsed = 0;
    const double update_rate = rate_param == info_.hardware_parameters.end() ?
      100.0 : std::stod(rate_param->second, &parsed);
    if (!std::isfinite(update_rate) || update_rate <= 0.0 ||
        (rate_param != info_.hardware_parameters.end() && parsed != rate_param->second.size()))
    {
      throw std::invalid_argument("update_rate must be a finite positive frequency in Hz");
    }
    const auto budget = std::chrono::duration<double>(WRITE_RETRY_BUDGET_RATIO / update_rate);
    if (!std::isfinite(budget.count()) ||
        budget >= std::chrono::steady_clock::duration::max())
    {
      throw std::invalid_argument("update_rate produces an unrepresentable I/O budget");
    }
    write_retry_budget_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(budget);
    if (write_retry_budget_ <= std::chrono::steady_clock::duration::zero()) {
      throw std::invalid_argument("update_rate produces an unrepresentable I/O budget");
    }
    config.port = info_.hardware_parameters.count("port")
      ? info_.hardware_parameters.at("port") : "/dev/ttyACM0";
    config.calibration_file = info_.hardware_parameters.count("calib_file")
      ? info_.hardware_parameters.at("calib_file") : "";
    // Simulated transport: the SDK fake bus stands in for the serial port, so
    // the production controller stack runs headless (runtime conformance).
    config.simulated = info_.hardware_parameters.count("simulated") &&
      is_truthy(info_.hardware_parameters.at("simulated"));

    // The URDF joint list is the canonical order; drop the SDK's default
    // "1".."6" so it is never appended to (duplicate motor ids otherwise).
    config.joint_order.clear();
    for (const auto & joint : info_.joints) {
      joint_names_.push_back(joint.name);
      config.joint_order.push_back(joint.name);
      const auto opening = joint.parameters.find("gripper_opening_direction");
      if (opening != joint.parameters.end()) {
        if (opening->second != "1" && opening->second != "-1") {
          throw std::invalid_argument("gripper_opening_direction must be 1 or -1");
        }
        gripper_opening_directions_[joint.name] = std::stoi(opening->second);
      }
      if (joint.parameters.count("id")) {
        const auto & text = joint.parameters.at("id");
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
          throw std::invalid_argument("invalid motor id for joint: " + joint.name);
        }
        config.motor_ids[joint.name] = std::stoi(text);
      }
      if (joint.command_interfaces.size() != 1 ||
          joint.command_interfaces.front().name != hardware_interface::HW_IF_POSITION ||
          joint.state_interfaces.size() != 2 ||
          std::none_of(joint.state_interfaces.begin(), joint.state_interfaces.end(),
            [](const auto & entry) { return entry.name == hardware_interface::HW_IF_POSITION; }) ||
          std::none_of(joint.state_interfaces.begin(), joint.state_interfaces.end(),
            [](const auto & entry) { return entry.name == hardware_interface::HW_IF_VELOCITY; }))
      {
        throw std::invalid_argument(
          "expected position command and position/velocity states: " + joint.name);
      }
    }

    // Legacy explicit startup-motion override, not semantic HOME.
    const std::string reset_str = info_.hardware_parameters.count("reset_positions")
      ? info_.hardware_parameters.at("reset_positions") : "";
    if (!reset_str.empty() && reset_str != "''" && reset_str != "\"\"") {
      auto reset_json = nlohmann::json::parse(reset_str);
      if (!reset_json.is_object()) {
        throw std::invalid_argument("reset_positions must be a joint-keyed object");
      }
      for (auto entry = reset_json.begin(); entry != reset_json.end(); ++entry) {
        config.reset_positions[entry.key()] = entry.value().get<double>();
      }
    }
    // One overload event retains its age across clear samples. A reported
    // overload past the bound fails; a clear held gripper does not expire.
    protection_timeout_ = std::chrono::seconds(1);
    const auto timeout_param = info_.hardware_parameters.find("protection_timeout");
    const std::string timeout_str =
      timeout_param == info_.hardware_parameters.end() ? "" : timeout_param->second;
    if (!timeout_str.empty() && timeout_str != "''" && timeout_str != "\"\"") {
      std::size_t timeout_parsed = 0;
      const double timeout_seconds = std::stod(timeout_str, &timeout_parsed);
      if (!std::isfinite(timeout_seconds) || timeout_seconds <= 0.0 ||
        timeout_seconds >= std::chrono::duration<double>(
          std::chrono::steady_clock::duration::max()).count() ||
        timeout_parsed != timeout_str.size())
      {
        throw std::invalid_argument("protection_timeout must be a positive duration in seconds");
      }
      protection_timeout_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(timeout_seconds));
      if (protection_timeout_ <= std::chrono::steady_clock::duration::zero()) {
        throw std::invalid_argument("protection_timeout is below clock resolution");
      }
    }
    config.motor_ids = so101::resolve_motor_ids(config.joint_order, config.motor_ids);
    arm_ = std::make_unique<so101::Arm>(config);
    arm_config_ = config;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Invalid hardware configuration: %s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  hw_positions_.resize(info_.joints.size(), 0.0);
  hw_velocities_.resize(info_.joints.size(), 0.0);
  hw_currents_.resize(info_.joints.size(), 0.0);
  hw_commands_.resize(info_.joints.size(), 0.0);
  for (const auto & name : joint_names_) {
    command_map_[name] = 0.0;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Configuring...");
  if (bus_ && bus_->is_open() &&
      arm_->health().lifecycle == so101::Lifecycle::Connected)
  {
    return hardware_interface::CallbackReturn::SUCCESS;
  }
  if (!arm_->deactivate()) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  try {
    const auto calibration = so101::Calibration::load(
      arm_config_.calibration_file, joint_names_, arm_config_.motor_ids);
    feetech::BusOptions options;
    options.port = arm_config_.port;
    options.baudrate = arm_config_.baudrate;
    options.simulated = arm_config_.simulated;
    for (const auto & name : joint_names_) {
      const auto & joint = calibration.at(name);
      feetech::MotorConfig motor;
      motor.id = static_cast<std::uint8_t>(arm_config_.motor_ids.at(name));
      motor.name = name;
      motor.homing_offset = joint.homing_offset;
      motor.range_min = joint.range_min;
      motor.range_max = joint.range_max;
      options.motors.push_back(motor);
    }
    bus_ = std::make_unique<feetech::Bus>(options);
    if (!bus_->open()) {
      throw std::runtime_error("cannot open port: " + options.port);
    }
    if (!arm_->attach_shared_bus(*bus_, calibration)) {
      bus_->close();
      throw std::runtime_error("cannot attach arm to hardware bus");
    }
  } catch (const so101::CalibError & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Failed to configure: %s: %s (joint %s)", e.what(), e.path.c_str(), e.joint.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Failed to configure: %s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }
  calibrated_ranges_ = arm_->calibrated_ranges();
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Configured!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
SO101SystemHardware::export_state_interfaces()
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

std::vector<hardware_interface::CommandInterface>
SO101SystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    command_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_activate(const rclcpp_lifecycle::State & previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Activating...");
  if ((!bus_ || !bus_->is_open()) &&
      on_configure(previous_state) != hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto probe = bus_->ping_all();
  if (!probe.ok) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Motor ID %u probe failed: %s",
      static_cast<unsigned>(probe.failed_id), probe.detail.c_str());
    if (!arm_->stop(so101::StopPolicy::TorqueOff)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Activation abort: torque release unconfirmed: %s", arm_->health().detail.c_str());
    }
    return hardware_interface::CallbackReturn::ERROR;
  }
  so101::ArmState preflight;
  if (!arm_->read(preflight) || !validate_feedback(preflight)) {
    motion_faulted_ = true;
    feedback_valid_ = false;
    arm_->stop(so101::StopPolicy::TorqueOff);
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (!arm_->activate()) {
    const auto & health = arm_->health();
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Activation failed: %s (fault=%d)", health.detail.c_str(),
      static_cast<int>(health.fault));
    return hardware_interface::CallbackReturn::ERROR;
  }

  so101::ArmState state;
  if (!arm_->read(state) || !validate_feedback(state)) {
    arm_->stop(so101::StopPolicy::TorqueOff);
    return hardware_interface::CallbackReturn::ERROR;
  }
  motion_faulted_ = false;
  feedback_valid_ = true;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto & reading = state.joints.at(joint_names_[i]);
    hw_positions_[i] = reading.position;
    hw_velocities_[i] = reading.velocity;
    hw_currents_[i] = reading.effort;
    hw_commands_[i] = arm_->command_targets().at(joint_names_[i]);
    command_map_[joint_names_[i]] = hw_commands_[i];
  }

  current_node_ = rclcpp::Node::make_shared("so101_joint_current_publisher");
  first_read_failure_.reset();
  first_write_failure_.reset();
  // Fresh protection session: the pose captured by the previous session must
  // not leak into this one.
  protection_.clear();
  for (const auto & name : joint_names_) {
    const auto & reading = state.joints.at(name);
    track_protection(name, reading.protection, reading.position, std::chrono::steady_clock::now());
  }
  current_pub_ =
    current_node_->create_publisher<ibrobot_msgs::msg::JointCurrent>(
    "/so101_follower/joint_currents", 10);

  RCLCPP_INFO(
    rclcpp::get_logger("SO101SystemHardware"),
    "Activated! Control loop running.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  // inactive -> active re-enters on_activate() without on_configure(), so the
  // bus must stay open: release torque only (runtime TORQUE_OFF stop).
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Deactivating (torque off, bus kept)...");
  if (!arm_->stop(so101::StopPolicy::TorqueOff)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Torque release failed: %s", arm_->health().detail.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }
  current_pub_.reset();
  current_node_.reset();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Cleaning up (closing bus)...");
  const bool ok = arm_->deactivate();
  if (bus_) {
    bus_->close();
  }
  return ok ? hardware_interface::CallbackReturn::SUCCESS :
         hardware_interface::CallbackReturn::ERROR;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_shutdown(const rclcpp_lifecycle::State &)
{
  const bool ok = !arm_ || arm_->deactivate();
  if (bus_) {
    bus_->close();
  }
  current_pub_.reset();
  current_node_.reset();
  return ok ? hardware_interface::CallbackReturn::SUCCESS :
         hardware_interface::CallbackReturn::ERROR;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_error(const rclcpp_lifecycle::State & state)
{
  return on_shutdown(state);
}

hardware_interface::return_type SO101SystemHardware::perform_command_mode_switch(
  const std::vector<std::string> &, const std::vector<std::string> & stop_interfaces)
{
  bool stopping = false;
  for (const auto & name : joint_names_) {
    if (std::find(stop_interfaces.begin(), stop_interfaces.end(), name + "/position") !=
        stop_interfaces.end())
    {
      stopping = true;
    }
  }
  if (!stopping) {
    return hardware_interface::return_type::OK;
  }
  // Controller deactivation is runtime HOLD: discard the previous target,
  // including a goal the motor has not reached, and hold fresh feedback.
  so101::ArmState state;
  if (motion_faulted_ || !arm_->read(state) || !validate_feedback(state)) {
    motion_faulted_ = true;
    feedback_valid_ = false;
    arm_->stop(so101::StopPolicy::TorqueOff);
    return hardware_interface::return_type::ERROR;
  }
  const auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto & name = joint_names_[i];
    const auto & reading = state.joints.at(name);
    store_reading(i, reading);
    track_protection(name, reading.protection, reading.position, now);
  }
  if (handle_active_protections(now) != hardware_interface::return_type::OK) {
    arm_->stop(so101::StopPolicy::TorqueOff);
    return hardware_interface::return_type::ERROR;
  }
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto & name = joint_names_[i];
    const auto held = protection_.find(name);
    hw_commands_[i] = held == protection_.end() ?
      state.joints.at(name).position : held->second.hold_position;
    command_map_[name] = hw_commands_[i];
  }
  feedback_valid_ = true;
  return arm_->write_targets(command_map_) ? hardware_interface::return_type::OK :
         hardware_interface::return_type::ERROR;
}

hardware_interface::return_type
SO101SystemHardware::read(const rclcpp::Time & time, const rclcpp::Duration &)
{
  so101::ArmState state;
  bool ok = false;
  const auto started = std::chrono::steady_clock::now();
  const auto budget = bus_ ? bus_->sync_read_timeout(joint_names_.size()) :
    std::chrono::milliseconds::zero();
  const auto deadline = started + budget;
  if (bus_) {
    const ScopedIoDeadline io_deadline(*bus_, deadline);
    // Like LeRobot's default sync_read(num_retry=0), allow one full receive
    // window. The control period must not truncate the USB latency allowance.
    ok = arm_->read(state);
  }

  if (!ok) {
    return handle_read_failure(started, budget);
  }

  first_read_failure_.reset();
  if (motion_faulted_ || !validate_feedback(state)) {
    motion_faulted_ = true;
    feedback_valid_ = false;
    return hardware_interface::return_type::ERROR;
  }
  feedback_valid_ = true;
  const auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < joint_names_.size(); i++) {
    const auto it = state.joints.find(joint_names_[i]);
    if (it == state.joints.end()) {
      return hardware_interface::return_type::ERROR;
    }
    store_reading(i, it->second);
    track_protection(
      joint_names_[i], it->second.protection, hw_positions_[i], now);
  }

  const auto protections = handle_active_protections(now);
  if (protections != hardware_interface::return_type::OK) {
    return protections;
  }
  publish_currents(time);
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
SO101SystemHardware::handle_read_failure(
  std::chrono::steady_clock::time_point started,
  std::chrono::steady_clock::duration budget)
{
  feedback_valid_ = false;
  if (!first_read_failure_) {
    first_read_failure_ = started;
  }
  const auto elapsed = std::chrono::steady_clock::now() - *first_read_failure_;
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
  const auto & health = arm_->health();
  if (elapsed >= FAILURE_TOLERANCE) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Arm read failing for %.1f ms (fault=%d: %s); reporting ERROR",
      std::chrono::duration<double, std::milli>(elapsed).count(),
      static_cast<int>(health.fault), health.detail.c_str());
    return hardware_interface::return_type::ERROR;
  }
  // Transient failure: hold last known values, do not crash the chain.
  RCLCPP_WARN_THROTTLE(
    rclcpp::get_logger("SO101SystemHardware"),
    steady_clock, 500,
    "Arm read failed (fault=%d: %s; elapsed=%.2f ms, budget=%.2f ms); holding last values",
    static_cast<int>(health.fault), health.detail.c_str(),
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
    std::chrono::duration<double, std::milli>(budget).count());
  return hardware_interface::return_type::OK;
}

void SO101SystemHardware::store_reading(size_t index, const so101::JointReading & reading)
{
  // A firmware protection does not fail the read (the frame decoded), but an
  // angle-sensor fault means position/velocity must not be trusted: keep the
  // last good values for that joint. Currents stay usable either way.
  if (!feetech::protection_invalidates_feedback(reading.protection)) {
    hw_positions_[index] = reading.position;
    hw_velocities_[index] = reading.velocity;
  }
  hw_currents_[index] = reading.effort;
}

bool SO101SystemHardware::validate_feedback(const so101::ArmState & state)
{
  for (const auto & name : joint_names_) {
    const auto found = state.joints.find(name);
    if (found == state.joints.end()) {
      RCLCPP_ERROR(rclcpp::get_logger("SO101SystemHardware"), "Missing feedback: %s", name.c_str());
      return false;
    }
    const auto & reading = found->second;
    const bool gripper_overload = gripper_opening_directions_.count(name) != 0 &&
      reading.protection == feetech::kProtectionOverload;
    if (!std::isfinite(reading.position) || !std::isfinite(reading.velocity) ||
      (reading.protection != 0 && !gripper_overload))
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Joint %s cannot continue position control: %s", name.c_str(),
        feetech::decode_protection_bits(reading.protection).c_str());
      return false;
    }
  }
  return true;
}

void SO101SystemHardware::track_protection(
  const std::string & joint, std::uint8_t bits, double pose,
  std::chrono::steady_clock::time_point now)
{
  const auto tracked = protection_.find(joint);
  if (tracked == protection_.end()) {
    if (bits == feetech::kProtectionOverload && gripper_opening_directions_.count(joint)) {
      protection_.emplace(joint, ProtectionTracking{bits, pose, now});
      RCLCPP_WARN(
        rclcpp::get_logger("SO101SystemHardware"),
        "Gripper overload hold ENTER: joint=%s, status=%s, hold_position=%.6f rad, "
        "opening_direction=%d, timeout=%.3f s",
        joint.c_str(), feetech::decode_protection_bits(bits).c_str(), pose,
        gripper_opening_directions_.at(joint),
        std::chrono::duration<double>(protection_timeout_).count());
    }
    return;
  }
  auto & event = tracked->second;
  const double age = std::chrono::duration<double>(now - event.first_seen).count();
  if (event.bits != bits) {
    RCLCPP_INFO(
      rclcpp::get_logger("SO101SystemHardware"),
      "Gripper overload hold STATUS: joint=%s, 0x%02x -> 0x%02x, "
      "event_age=%.3f s, opening_pending=%s",
      joint.c_str(), static_cast<unsigned>(event.bits), static_cast<unsigned>(bits),
      age, event.opening_pending ? "true" : "false");
  }
  event.bits = bits;
  const auto joint_it = std::find(joint_names_.begin(), joint_names_.end(), joint);
  const auto index = static_cast<size_t>(std::distance(joint_names_.begin(), joint_it));
  if (event.opening_pending && bits == 0 &&
    is_opening_target(joint, hw_commands_.at(index), event.hold_position))
  {
    // This read follows a successful opening write. It confirms only the
    // reported fault cleared, not target arrival or physical force safety.
    RCLCPP_INFO(
      rclcpp::get_logger("SO101SystemHardware"),
      "Gripper overload hold EXIT: joint=%s, feedback confirmed status=0 after "
      "opening write, target=%.6f rad, feedback=%.6f rad, event_age=%.3f s",
      joint.c_str(), event.opening_target, pose, age);
    protection_.erase(tracked);
  }
}

bool SO101SystemHardware::is_opening_target(
  const std::string & joint, double target, double hold) const
{
  if (!feedback_valid_ || !std::isfinite(target)) {
    return false;
  }
  const auto & limits = calibrated_ranges_.at(joint);
  const int target_tick = feetech::radians_to_ticks(
    std::clamp(target, limits.first, limits.second));
  const int hold_tick = feetech::radians_to_ticks(hold);
  return gripper_opening_directions_.at(joint) * (target_tick - hold_tick) > 0;
}

hardware_interface::return_type
SO101SystemHardware::handle_active_protections(std::chrono::steady_clock::time_point now)
{
  if (protection_.empty()) {
    return hardware_interface::return_type::OK;
  }
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
  std::string detail;
  bool expired = false;
  for (auto & entry : protection_) {
    const auto elapsed = now - entry.second.first_seen;
    if (!detail.empty()) {
      detail += "; ";
    }
    detail += "motor " + entry.first + ": ";
    if (entry.second.bits == 0) {
      detail += "hold active, firmware cleared";
      continue;
    }
    const auto protected_s = static_cast<std::int64_t>(
      std::chrono::duration<double>(elapsed).count());
    detail += feetech::decode_protection_bits(entry.second.bits) +
      " for " + std::to_string(protected_s) + " s";
    if (elapsed > protection_timeout_) {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Gripper overload hold TIMEOUT: joint=%s, status=0x%02x, event_age=%.3f s, "
        "timeout=%.3f s, opening_pending=%s",
        entry.first.c_str(), static_cast<unsigned>(entry.second.bits),
        std::chrono::duration<double>(elapsed).count(),
        std::chrono::duration<double>(protection_timeout_).count(),
        entry.second.opening_pending ? "true" : "false");
      expired = true;
    }
  }
  if (expired) {
    motion_faulted_ = true;
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Motor protection still reported after %.1f s (%s); reporting ERROR",
      std::chrono::duration<double>(protection_timeout_).count(), detail.c_str());
    return hardware_interface::return_type::ERROR;
  }
  RCLCPP_WARN_THROTTLE(
    rclcpp::get_logger("SO101SystemHardware"), steady_clock, 2000,
    "Gripper position hold active (%s); force is not regulated", detail.c_str());
  return hardware_interface::return_type::OK;
}
hardware_interface::return_type
SO101SystemHardware::write(const rclcpp::Time &, const rclcpp::Duration & period)
{
  if (motion_faulted_) {
    return hardware_interface::return_type::ERROR;
  }
  // Humble may keep invoking I/O for inactive hardware. Read feedback there,
  // but do not send position targets or turn an intentional torque-off into
  // a hardware error. Activation will discard these command buffers.
  const auto lifecycle = arm_->health().lifecycle;
  if (lifecycle == so101::Lifecycle::Connected ||
      lifecycle == so101::Lifecycle::Disconnected || lifecycle == so101::Lifecycle::Faulted)
  {
    static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("SO101SystemHardware"), steady_clock, 500,
      "Skipping arm write in lifecycle %d; read path monitors sustained failures",
      static_cast<int>(lifecycle));
    first_write_failure_.reset();
    return hardware_interface::return_type::OK;
  }
  for (size_t i = 0; i < joint_names_.size(); i++) {
    const auto & name = joint_names_[i];
    const auto tracked = protection_.find(name);
    if (tracked != protection_.end()) {
      // Only an opening command may override the hold. Do not infer force
      // limiting from position holding or from a cleared firmware status.
      if (is_opening_target(name, hw_commands_[i], tracked->second.hold_position)) {
        command_map_[name] = hw_commands_[i];
        continue;
      }
      command_map_[name] = tracked->second.hold_position;
    } else {
      command_map_[name] = hw_commands_[i];
    }
  }
  bool write_ok = false;
  const auto started = std::chrono::steady_clock::now();
  const auto budget = period.nanoseconds() > 0 ? std::min(
    write_retry_budget_, std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(period.seconds() * WRITE_RETRY_BUDGET_RATIO))) :
    write_retry_budget_;
  const auto deadline = started + budget;
  if (bus_) {
    const ScopedIoDeadline io_deadline(*bus_, deadline);
    do {
      write_ok = arm_->write_targets(command_map_);
      if (!write_ok) {
        std::this_thread::sleep_until(std::min(
          deadline, std::chrono::steady_clock::now() + std::chrono::microseconds(500)));
      }
    } while (!write_ok && std::chrono::steady_clock::now() < deadline);
  }
  if (!write_ok) {
    if (!first_write_failure_) {
      first_write_failure_ = started;
    }
    const auto elapsed = std::chrono::steady_clock::now() - *first_write_failure_;
    static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    const auto & health = arm_->health();
    if (elapsed >= FAILURE_TOLERANCE) {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Arm write failing for %.1f ms (fault=%d: %s); reporting ERROR",
        std::chrono::duration<double, std::milli>(elapsed).count(),
        static_cast<int>(health.fault), health.detail.c_str());
      return hardware_interface::return_type::ERROR;
    }
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("SO101SystemHardware"),
      steady_clock, 500, "Arm write failed (fault=%d: %s)",
      static_cast<int>(health.fault), health.detail.c_str());
    return hardware_interface::return_type::OK;
  }
  // A successful send is not confirmation of recovery. Keep the same event
  // and deadline until a later valid read confirms a cleared status.
  for (auto & [name, event] : protection_) {
    const double target = command_map_.at(name);
    const bool opening = is_opening_target(name, target, event.hold_position);
    if (opening && !event.opening_pending) {
      RCLCPP_INFO(
        rclcpp::get_logger("SO101SystemHardware"),
        "Gripper overload hold OPENING_SENT: joint=%s, target=%.6f rad, "
        "hold_position=%.6f rad, status=0x%02x; awaiting feedback, deadline retained",
        name.c_str(), target, event.hold_position, static_cast<unsigned>(event.bits));
    } else if (!opening && event.opening_pending) {
      RCLCPP_INFO(
        rclcpp::get_logger("SO101SystemHardware"),
        "Gripper overload hold OPENING_CANCELLED: joint=%s; holding, deadline retained",
        name.c_str());
    }
    event.opening_pending = opening;
    if (opening) {
      event.opening_target = target;
    }
  }
  first_write_failure_.reset();
  return hardware_interface::return_type::OK;
}

void SO101SystemHardware::publish_currents(const rclcpp::Time & stamp)
{
  if (!current_pub_) {
    return;
  }
  ibrobot_msgs::msg::JointCurrent msg;
  msg.header.stamp = stamp;
  msg.name.reserve(joint_names_.size());
  msg.current.reserve(hw_currents_.size());
  for (size_t i = 0; i < joint_names_.size(); i++) {
    msg.name.push_back(joint_names_[i]);
    msg.current.push_back(hw_currents_[i]);
  }
  current_pub_->publish(msg);
}

}  // namespace so101_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  so101_hardware::SO101SystemHardware,
  hardware_interface::SystemInterface)
