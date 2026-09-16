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

#include "so101/calibration.hpp"

#include <fstream>
#include <limits>
#include <set>
#include <nlohmann/json.hpp>

namespace so101
{

namespace
{

int require_int(
  const nlohmann::json & entry, const char * field, const std::string & path,
  const std::string & joint)
{
  if (!entry.contains(field) || !entry[field].is_number_integer()) {
    throw CalibError(
      std::string("calibration entry is missing integer field '") + field + "'",
      path, joint);
  }
  if (entry[field] < std::numeric_limits<int>::min() ||
      entry[field] > std::numeric_limits<int>::max())
  {
    throw CalibError(std::string("integer overflow in '") + field + "'", path, joint);
  }
  return entry[field].get<int>();
}

}  // namespace

Calibration Calibration::load(
  const std::string & path, const std::vector<std::string> & joint_order,
  const std::map<std::string, int> & motor_ids)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    throw CalibError("calibration file cannot be opened", path);
  }
  nlohmann::json data;
  try {
    data = nlohmann::json::parse(file);
  } catch (const nlohmann::json::exception & e) {
    throw CalibError(std::string("invalid calibration JSON: ") + e.what(), path);
  }

  if (!data.is_object()) {
    throw CalibError("calibration file root must be a JSON object", path);
  }

  // LeKiwi-format compatibility: entries may carry the motor id in an "id"
  // field with arbitrary keys, instead of the SO-101 convention of using
  // the motor id as the key itself.
  std::map<int, const nlohmann::json *> by_id_field;
  for (auto it = data.begin(); it != data.end(); ++it) {
    if (it.value().is_object() && it.value().contains("id") &&
        it.value()["id"].is_number_integer())
    {
      const int id = require_int(it.value(), "id", path, it.key());
      if (id < 1 || id > 253 || !by_id_field.emplace(id, &it.value()).second) {
        throw CalibError("invalid or duplicate calibration motor id", path, it.key());
      }
    }
  }

  Calibration calibration;
  for (const std::string & joint : joint_order) {
    const std::string key = motor_ids.count(joint) ? std::to_string(motor_ids.at(joint)) : joint;
    const nlohmann::json * entry = nullptr;
    const auto direct = data.find(key);
    if (direct != data.end() && direct->is_object()) {
      entry = &direct.value();
    } else {
      try {
        const auto by_id = by_id_field.find(std::stoi(key));
        if (by_id != by_id_field.end()) {
          entry = by_id->second;
        }
      } catch (const std::exception &) {
        // Non-numeric joint names cannot match the id-field format.
      }
    }
    if (entry == nullptr) {
      throw CalibError("calibration entry is missing for joint", path, joint);
    }
    JointCalibration joint_calib;
    joint_calib.homing_offset = require_int(*entry, "homing_offset", path, joint);
    joint_calib.range_min = require_int(*entry, "range_min", path, joint);
    joint_calib.range_max = require_int(*entry, "range_max", path, joint);
    if (joint_calib.homing_offset < -2047 || joint_calib.homing_offset > 2047 ||
        joint_calib.range_min < 0 || joint_calib.range_max > 4095 ||
        joint_calib.range_min >= joint_calib.range_max)
    {
      throw CalibError("calibration offset or range is outside STS position limits", path, joint);
    }
    if (entry->contains("drive_mode")) {
      const auto & mode = (*entry)["drive_mode"];
      if (mode.is_boolean()) {
        joint_calib.drive_mode = mode.get<bool>();
      } else if (mode.is_number_integer() && (mode == 0 || mode == 1)) {
        joint_calib.drive_mode = mode == 1;
      } else {
        throw CalibError("drive_mode must be a boolean or 0/1", path, joint);
      }
    }
    calibration.joints_[joint] = joint_calib;
  }
  return calibration;
}

std::map<std::string, int> resolve_motor_ids(
  const std::vector<std::string> & joint_order, const std::map<std::string, int> & motor_ids)
{
  if (joint_order.empty()) {
    throw std::invalid_argument("joint_order must not be empty");
  }
  std::map<std::string, int> resolved;
  std::set<int> seen;
  for (const auto & joint : joint_order) {
    int id;
    if (motor_ids.count(joint)) {
      id = motor_ids.at(joint);
    } else {
      if (joint.empty() || joint.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("joint requires an explicit motor id: " + joint);
      }
      id = std::stoi(joint);
    }
    if (id < 1 || id > 253 || !seen.insert(id).second ||
        !resolved.emplace(joint, id).second)
    {
      throw std::invalid_argument("invalid or duplicate motor id for joint: " + joint);
    }
  }
  for (const auto & [joint, id] : motor_ids) {
    (void)id;
    if (!resolved.count(joint)) {
      throw std::invalid_argument("motor id mapping has unknown joint: " + joint);
    }
  }
  return resolved;
}

const JointCalibration & Calibration::at(const std::string & joint) const
{
  const auto it = joints_.find(joint);
  if (it == joints_.end()) {
    throw CalibError("joint is not present in the loaded calibration", "", joint);
  }
  return it->second;
}

}  // namespace so101
