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

#ifndef SO101__CALIBRATION_HPP_
#define SO101__CALIBRATION_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace so101
{

/// Calibration data for one joint, in raw servo units (never SI).
struct JointCalibration
{
  int homing_offset = 0;  // ticks
  int range_min = 0;      // ticks
  int range_max = 4095;   // ticks
  bool drive_mode = false;  // invert normalized gripper travel
};

/// Calibration load/validation failure. Carries the file path and, when the
/// failure is about a specific joint, the joint name.
class CalibError : public std::runtime_error
{
public:
  CalibError(std::string message, std::string path, std::string joint = "")
  : std::runtime_error(std::move(message)), path(std::move(path)), joint(std::move(joint))
  {
  }
  std::string path;
  std::string joint;
};

/// Calibration for an arm, keyed by canonical joint name.
///
/// Two on-disk layouts are accepted:
/// 1. SO-101 format: the motor id (joint name) is the JSON key.
/// 2. LeKiwi format: entries carry the motor id in an "id" field with
///    arbitrary keys; joints match by their numeric name.
class Calibration
{
public:
  /// Load and validate a calibration file against `joint_order`. Fails with
  /// CalibError when the file is missing, unreadable, not valid JSON, or
  /// missing any joint in `joint_order` (or an entry lacks a required field).
  /// Partial calibration data is never silently defaulted.
  static Calibration load(
    const std::string & path, const std::vector<std::string> & joint_order,
    const std::map<std::string, int> & motor_ids = {});

  const JointCalibration & at(const std::string & joint) const;

private:
  std::map<std::string, JointCalibration> joints_;
};

/// Resolve numeric names or explicit legacy URDF joint-to-motor mappings.
std::map<std::string, int> resolve_motor_ids(
  const std::vector<std::string> & joint_order, const std::map<std::string, int> & motor_ids);

}  // namespace so101

#endif  // SO101__CALIBRATION_HPP_
