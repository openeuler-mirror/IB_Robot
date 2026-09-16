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

#include "so101/leader.hpp"

#include <algorithm>
#include <stdexcept>

#include "feetech/conversion.hpp"

namespace so101
{

LeaderArm::LeaderArm(LeaderConfig config) : config_(std::move(config))
{
  config_.motor_ids = resolve_motor_ids(config_.joint_order, config_.motor_ids);
  for (const std::string & joint : config_.joint_order) {
    id_of_[joint] = static_cast<std::uint8_t>(config_.motor_ids.at(joint));
  }
  if (id_of_.find(config_.gripper_joint) == id_of_.end()) {
    throw std::invalid_argument(
      "gripper_joint '" + config_.gripper_joint + "' is not part of joint_order");
  }
}

bool LeaderArm::connect()
{
  bus_.reset();
  if (config_.calibration_version != 1) {
    health_.lifecycle = Lifecycle::Faulted;
    health_.fault = feetech::Fault::ConfigFailed;
    health_.detail = "read-only leader requires explicit calibration_version=1 and prior provisioning";
    return false;
  }
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

  feetech::BusOptions options;
  options.port = config_.port;
  options.baudrate = config_.baudrate;
  options.simulated = config_.simulated;
  for (const std::string & joint : config_.joint_order) {
    feetech::MotorConfig motor;
    motor.id = id_of_.at(joint);
    motor.name = joint;
    const auto & calib = calibration_.at(joint);
    motor.homing_offset = calib.homing_offset;
    motor.range_min = calib.range_min;
    motor.range_max = calib.range_max;
    options.motors.push_back(motor);
  }
  // Read-only: no motor configuration is applied and torque stays untouched.
  auto bus = std::make_unique<feetech::Bus>(std::move(options));
  if (!bus->open()) {
    health_.lifecycle = Lifecycle::Faulted;
    health_.fault = feetech::Fault::PortOpenFailed;
    health_.detail = "cannot open port: " + config_.port;
    return false;
  }
  const auto provisioned = bus->verify_calibration();
  if (!provisioned.ok) {
    health_.lifecycle = Lifecycle::Faulted;
    health_.fault = provisioned.fault;
    health_.detail = provisioned.detail + " (motor " + std::to_string(provisioned.failed_id) + ")";
    return false;
  }
  bus_ = std::move(bus);
  health_.lifecycle = Lifecycle::Activated;
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

void LeaderArm::disconnect()
{
  bus_.reset();
  health_ = ArmHealth{};
}

bool LeaderArm::read(ArmState & out)
{
  out.joints.clear();
  if (!bus_ || health_.lifecycle != Lifecycle::Activated) {
    return false;
  }
  std::vector<feetech::MotorSample> samples;
  const auto result = bus_->sync_read(samples);
  if (!result.ok) {
    health_.fault = result.fault;
    health_.detail = result.detail;
    return false;
  }

  ArmState state;
  state.stamp = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < config_.joint_order.size(); ++i) {
    const std::string & joint = config_.joint_order[i];
    JointReading reading;
    reading.velocity = samples[i].velocity;
    reading.effort = samples[i].effort;
    if (joint == config_.gripper_joint) {
      // Gripper: normalized opening in [0, 1], drive_mode-aware. The
      // calibrated range is in absolute ticks; the centered radian reading is
      // converted back to raw ticks for normalization (leader_arm.py
      // semantics).
      const JointCalibration & calib = calibration_.at(joint);
      const double raw =
        samples[i].position * feetech::kTicksPerRad + static_cast<double>(feetech::kCenterTick);
      const double raw_min = static_cast<double>(calib.range_min);
      const double raw_max = static_cast<double>(calib.range_max);
      const double bounded = std::clamp(raw, raw_min, raw_max);
      double normalized = (bounded - raw_min) / std::max(1.0, raw_max - raw_min);
      if (calib.drive_mode) {
        normalized = 1.0 - normalized;
      }
      reading.position = normalized;
    } else {
      reading.position = samples[i].position;
    }
    state.joints[joint] = reading;
  }
  out = std::move(state);
  health_.fault = feetech::Fault::None;
  health_.detail.clear();
  return true;
}

const ArmHealth & LeaderArm::health() const { return health_; }

feetech::Bus::SimControl & LeaderArm::sim()
{
  if (!bus_) {
    throw std::logic_error("LeaderArm::sim() called before connect()");
  }
  return bus_->sim();
}

}  // namespace so101
