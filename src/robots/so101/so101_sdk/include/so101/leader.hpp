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

#ifndef SO101__LEADER_HPP_
#define SO101__LEADER_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "feetech/bus.hpp"
#include "so101/arm.hpp"
#include "so101/calibration.hpp"

namespace so101
{

/// Leader arm construction options.
struct LeaderConfig
{
  std::string port;
  std::uint32_t baudrate = 1'000'000;
  std::string calibration_file;
  /// Explicit schema acknowledgement for an already provisioned legacy JSON.
  /// Must be 1; zero (undeclared) is rejected before opening the bus.
  int calibration_version = 0;
  /// Canonical joint order (the last joint is the gripper by convention).
  std::vector<std::string> joint_order = {"1", "2", "3", "4", "5", "6"};
  std::map<std::string, int> motor_ids;
  /// Gripper joint name (must be part of joint_order).
  std::string gripper_joint = "6";
  bool simulated = false;
};

/// Read-only SO-101 leader arm. Exposes state reading only — there is no
/// command surface, and connecting a leader never torque-enables
/// follower-style control. Arm joints read in radians (centered on the
/// calibration zero); the gripper reads as a normalized [0, 1] opening.
class LeaderArm
{
public:
  explicit LeaderArm(LeaderConfig config);

  LeaderArm(const LeaderArm &) = delete;
  LeaderArm & operator=(const LeaderArm &) = delete;

  /// Open the bus and load/validate the calibration. No motor configuration
  /// or torque control is ever performed.
  bool connect();
  void disconnect();

  /// Read joint targets for the follower. Arm joints are radians; the
  /// gripper joint is normalized opening in [0, 1] (drive_mode-aware).
  /// On failure `out` is cleared and false is returned.
  bool read(ArmState & out);

  const ArmHealth & health() const;

  /// Simulated-transport control handle (aborts when not simulated).
  feetech::Bus::SimControl & sim();

private:
  LeaderConfig config_;
  std::unique_ptr<feetech::Bus> bus_;
  Calibration calibration_;
  ArmHealth health_;
  std::map<std::string, std::uint8_t> id_of_;
};

}  // namespace so101

#endif  // SO101__LEADER_HPP_
