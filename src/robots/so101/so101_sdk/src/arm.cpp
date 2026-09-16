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

#include "so101/arm.hpp"

#include <stdexcept>
#include <cmath>

#include "feetech/conversion.hpp"

namespace so101
{

namespace
{

feetech::BusOptions make_bus_options(
  const ArmConfig & config, const Calibration & calibration)
{
  feetech::BusOptions options;
  options.port = config.port;
  options.baudrate = config.baudrate;
  options.simulated = config.simulated;
  for (const std::string & joint : config.joint_order) {
    feetech::MotorConfig motor;
    motor.id = static_cast<std::uint8_t>(config.motor_ids.at(joint));
    motor.name = joint;
    const JointCalibration & joint_calib = calibration.at(joint);
    motor.homing_offset = joint_calib.homing_offset;
    motor.range_min = joint_calib.range_min;
    motor.range_max = joint_calib.range_max;
    options.motors.push_back(motor);
  }
  return options;
}

}  // namespace

Arm::Arm(ArmConfig config) : config_(std::move(config))
{
  config_.motor_ids = resolve_motor_ids(config_.joint_order, config_.motor_ids);
  for (const std::string & joint : config_.joint_order) {
    const std::uint8_t id = static_cast<std::uint8_t>(config_.motor_ids.at(joint));
    id_of_[joint] = id;
    my_ids_.push_back(id);  // joint_order order: scoped reads map by index
  }
  for (const auto & [joint, position] : config_.reset_positions) {
    if (!id_of_.count(joint) || !std::isfinite(position)) {
      throw std::invalid_argument("invalid reset_positions entry: " + joint);
    }
  }
}

Arm::~Arm() { deactivate(); }

bool Arm::connect()
{
  if (bus_ && (health_.lifecycle == Lifecycle::Connected ||
               health_.lifecycle == Lifecycle::Activated))
  {
    return true;
  }
  if (bus_ && !deactivate()) {
    return false;
  }
  owned_bus_.reset();
  bus_ = nullptr;
  owns_bus_ = false;
  Calibration calibration;
  try {
    calibration = Calibration::load(config_.calibration_file, config_.joint_order, config_.motor_ids);
  } catch (const CalibError & e) {
    health_.lifecycle = Lifecycle::Faulted;
    health_.fault = feetech::Fault::ConfigFailed;
    health_.detail = std::string(e.what()) + ": " + e.path +
                     (e.joint.empty() ? std::string() : " (joint " + e.joint + ")");
    return false;
  }
  calibration_ = calibration;

  auto bus = std::make_unique<feetech::Bus>(make_bus_options(config_, calibration_));
  if (!bus->open()) {
    health_.lifecycle = Lifecycle::Faulted;
    health_.fault = feetech::Fault::PortOpenFailed;
    health_.detail = "cannot open port: " + config_.port;
    return false;
  }
  owned_bus_ = std::move(bus);
  bus_ = owned_bus_.get();
  owns_bus_ = true;
  health_.lifecycle = Lifecycle::Connected;
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

bool Arm::attach_shared_bus(feetech::Bus & bus, const Calibration & calibration)
{
  if (bus_ && (health_.lifecycle == Lifecycle::Connected ||
               health_.lifecycle == Lifecycle::Activated))
  {
    return false;  // already attached/connected
  }
  owned_bus_.reset();
  calibration_ = calibration;
  bus_ = &bus;  // non-owning: the caller owns and closes this bus
  owns_bus_ = false;
  health_.lifecycle = Lifecycle::Connected;
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

bool Arm::activate()
{
  if (health_.lifecycle != Lifecycle::Connected || !bus_) {
    health_.detail = "activate requires connect() first";
    return false;
  }
  const auto abort = [this](feetech::Fault fault, std::string detail) {
      const auto release = bus_->emergency_release(my_ids_);
      health_.lifecycle = Lifecycle::Faulted;
      health_.fault = release.ok ? fault : release.fault;
      health_.detail = std::move(detail);
      if (!release.ok) {
        health_.detail += "; torque release unconfirmed: " + release.detail;
      }
      last_targets_.clear();
      return false;
    };
  const auto config_result = bus_->apply_configs(my_ids_);
  if (!config_result.ok) {
    return abort(config_result.fault, "activation failed during configuration: " + config_result.detail);
  }

  std::vector<feetech::MotorSample> samples;
  const auto sync_result = bus_->sync_read(samples, my_ids_);
  if (!sync_result.ok) {
    return abort(sync_result.fault, "activation failed during initial feedback sync: " + sync_result.detail);
  }

  // Explicit legacy startup targets remain supported; semantic HOME is not
  // part of SDK activation. Without overrides every joint holds its reading.
  last_targets_.clear();
  for (std::size_t i = 0; i < config_.joint_order.size(); ++i) {
    const std::string & joint = config_.joint_order[i];
    const auto reset = config_.reset_positions.find(joint);
    last_targets_[joint] =
      (reset != config_.reset_positions.end()) ? reset->second : samples[i].position;
  }
  if (!write_targets_unlocked(last_targets_)) {
    return abort(health_.fault, "activation failed while seeding initial commands: " + health_.detail);
  }
  health_.lifecycle = Lifecycle::Activated;
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

bool Arm::read(ArmState & out)
{
  out.joints.clear();
  if ((health_.lifecycle != Lifecycle::Activated && health_.lifecycle != Lifecycle::Connected) || !bus_) {
    return false;
  }
  std::vector<feetech::MotorSample> samples;
  const auto result = bus_->sync_read(samples, my_ids_);
  if (!result.ok) {
    health_.fault = result.fault;
    health_.detail = result.detail;
    return false;
  }
  ArmState state;
  state.stamp = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < config_.joint_order.size(); ++i) {
    JointReading reading;
    reading.position = samples[i].position;
    reading.velocity = samples[i].velocity;
    reading.effort = samples[i].effort;
    state.joints[config_.joint_order[i]] = reading;
  }
  out = std::move(state);
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

bool Arm::write_targets_unlocked(const std::map<std::string, double> & targets)
{
  std::vector<feetech::MotorTarget> motor_targets;
  for (const auto & [joint, position] : targets) {
    const auto id = id_of_.find(joint);
    if (id == id_of_.end()) {
      health_.fault = feetech::Fault::WriteRejected;
      health_.detail = "unknown joint in write_targets: " + joint;
      return false;
    }
    feetech::MotorTarget target;
    target.id = id->second;
    target.position = position;
    motor_targets.push_back(target);
  }
  const auto result = bus_->sync_write_positions(motor_targets);
  health_.fault = result.fault;
  health_.detail = result.detail;
  return result.ok;
}

bool Arm::write_targets(const std::map<std::string, double> & targets)
{
  if (health_.lifecycle != Lifecycle::Activated || !bus_) {
    return false;
  }
  if (!write_targets_unlocked(targets)) {
    return false;
  }
  for (const auto & [joint, position] : targets) {
    last_targets_[joint] = position;
  }
  return true;
}

bool Arm::hold()
{
  return bus_ && health_.lifecycle == Lifecycle::Activated &&
         write_targets_unlocked(last_targets_);
}

bool Arm::stop(StopPolicy policy)
{
  if (!bus_) {
    return false;
  }
  if (policy == StopPolicy::TorqueOff) {
    // Torque released: the arm is connected but no longer activated, so a
    // later activate() re-applies motor configuration and re-seeds targets.
    const auto result = bus_->emergency_release(my_ids_);
    const bool ok = result.ok;
    health_.fault = result.fault;
    health_.detail = result.detail;
    last_targets_.clear();
    if (!ok) {
      health_.lifecycle = Lifecycle::Faulted;
    }
    if (ok && health_.lifecycle == Lifecycle::Activated) {
      health_.lifecycle = Lifecycle::Connected;
    }
    return ok;
  }
  return hold();
}

bool Arm::deactivate()
{
  if (!bus_) {
    return true;
  }
  const auto result = bus_->emergency_release(my_ids_);
  if (owns_bus_) {
    bus_->close();
    owned_bus_.reset();
  }
  bus_ = nullptr;
  owns_bus_ = false;
  last_targets_.clear();
  health_.lifecycle = result.ok ? Lifecycle::Disconnected : Lifecycle::Faulted;
  health_.fault = result.fault;
  health_.detail = result.detail;
  return result.ok;
}

std::map<std::string, std::pair<double, double>> Arm::calibrated_ranges() const
{
  std::map<std::string, std::pair<double, double>> ranges;
  for (const std::string & joint : config_.joint_order) {
    const JointCalibration & joint_calib = calibration_.at(joint);
    ranges[joint] = {
      feetech::ticks_to_radians(joint_calib.range_min),
      feetech::ticks_to_radians(joint_calib.range_max)};
  }
  return ranges;
}

const ArmHealth & Arm::health() const { return health_; }

const std::vector<std::string> & Arm::joint_names() const { return config_.joint_order; }

feetech::Bus::SimControl & Arm::sim()
{
  if (!bus_) {
    throw std::logic_error("Arm::sim() called before connect()");
  }
  return bus_->sim();
}

}  // namespace so101
