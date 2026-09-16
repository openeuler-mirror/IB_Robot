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

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "so101/calibration.hpp"

namespace
{

class TempCalibFile
{
public:
  explicit TempCalibFile(const std::string & contents)
  {
    path_ = "/tmp/so101_sdk_calib_test_" + std::to_string(getpid()) + ".json";
    std::ofstream file(path_);
    file << contents;
  }
  ~TempCalibFile() { std::remove(path_.c_str()); }
  const std::string & path() const { return path_; }

private:
  std::string path_;
};

const std::vector<std::string> kJoints = {"1", "2", "3", "4", "5", "6"};

std::string valid_calib_json()
{
  return R"({
    "1": {"homing_offset": -10, "range_min": 100, "range_max": 4000},
    "2": {"homing_offset": 20, "range_min": 120, "range_max": 3900},
    "3": {"homing_offset": 0, "range_min": 150, "range_max": 3800},
    "4": {"homing_offset": 5, "range_min": 200, "range_max": 3700},
    "5": {"homing_offset": -7, "range_min": 250, "range_max": 3600},
    "6": {"homing_offset": 0, "range_min": 300, "range_max": 3500, "drive_mode": true}
  })";
}

TEST(Calibration, LoadsAllJoints)
{
  TempCalibFile file(valid_calib_json());
  const auto calib = so101::Calibration::load(file.path(), kJoints);
  EXPECT_EQ(calib.at("1").homing_offset, -10);
  EXPECT_EQ(calib.at("2").range_min, 120);
  EXPECT_TRUE(calib.at("6").drive_mode);
  EXPECT_FALSE(calib.at("1").drive_mode);
}

TEST(Calibration, MissingFileFailsWithPath)
{
  try {
    so101::Calibration::load("/tmp/so101_sdk_does_not_exist.json", kJoints);
    FAIL() << "expected CalibError";
  } catch (const so101::CalibError & e) {
    EXPECT_NE(e.path.find("so101_sdk_does_not_exist"), std::string::npos);
  }
}

TEST(Calibration, MissingJointFailsWithJointName)
{
  // Calibration for joints 1-5 only; joint 6 is absent.
  TempCalibFile file(R"({
    "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
  })");
  try {
    so101::Calibration::load(file.path(), kJoints);
    FAIL() << "expected CalibError";
  } catch (const so101::CalibError & e) {
    EXPECT_EQ(e.joint, "6");
    EXPECT_NE(e.path.find(file.path()), std::string::npos);
  }
}

TEST(Calibration, IncompleteEntryFailsWithoutDefaulting)
{
  // Joint 3 is missing range_max: no silent defaults are allowed.
  TempCalibFile file(R"({
    "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "3": {"homing_offset": 0, "range_min": 0},
    "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "6": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
  })");
  try {
    so101::Calibration::load(file.path(), kJoints);
    FAIL() << "expected CalibError";
  } catch (const so101::CalibError & e) {
    EXPECT_EQ(e.joint, "3");
    EXPECT_NE(std::string(e.what()).find("range_max"), std::string::npos);
  }
}

TEST(Calibration, LeKiwiIdFieldFormatAccepted)
{
  // LeKiwi layout: entries carry the motor id in an "id" field under
  // arbitrary keys (e.g. semantic names).
  TempCalibFile file(R"({
    "shoulder_pan": {"id": 1, "homing_offset": -11, "range_min": 111, "range_max": 4011},
    "shoulder_lift": {"id": 2, "homing_offset": 22, "range_min": 122, "range_max": 4022},
    "elbow": {"id": 3, "homing_offset": 0, "range_min": 150, "range_max": 3800},
    "wrist": {"id": 4, "homing_offset": 0, "range_min": 200, "range_max": 3700},
    "roll": {"id": 5, "homing_offset": 0, "range_min": 250, "range_max": 3600},
    "gripper": {"id": 6, "homing_offset": 0, "range_min": 300, "range_max": 3500}
  })");
  const auto calib = so101::Calibration::load(file.path(), kJoints);
  EXPECT_EQ(calib.at("1").homing_offset, -11);
  EXPECT_EQ(calib.at("2").range_max, 4022);
  EXPECT_EQ(calib.at("6").range_min, 300);
}

TEST(Calibration, MissingJointInBothFormatsFails)
{
  // Neither a direct "7" key nor an entry with id=7 exists.
  TempCalibFile file(R"({
    "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
    "arm_wrist": {"id": 9, "homing_offset": 0, "range_min": 0, "range_max": 4095}
  })");
  try {
    so101::Calibration::load(file.path(), kJoints);
    FAIL() << "expected CalibError";
  } catch (const so101::CalibError & e) {
    EXPECT_EQ(e.joint, "6");
  }
}

TEST(Calibration, MalformedJsonReportsCalibErrorWithPath)
{
  TempCalibFile file("{not json");
  try {
    so101::Calibration::load(file.path(), {"1"});
    FAIL();
  } catch (const so101::CalibError & error) {
    EXPECT_EQ(error.path, file.path());
  }
}

TEST(Calibration, RejectsInvalidRangesOffsetsAndDriveModes)
{
  for (const auto & fields : {
      R"("homing_offset":2048,"range_min":0,"range_max":4095)",
      R"("homing_offset":0,"range_min":42,"range_max":42)",
      R"("homing_offset":0,"range_min":0,"range_max":4096)",
      R"("homing_offset":4294967296,"range_min":0,"range_max":4095)",
      R"("homing_offset":0,"range_min":0,"range_max":4095,"drive_mode":2)"})
  {
    TempCalibFile file(std::string("{\"1\":{") + fields + "}}");
    EXPECT_THROW(so101::Calibration::load(file.path(), {"1"}), so101::CalibError);
  }
}

TEST(Calibration, ExplicitMappedJointUsesNumericCalibrationAndIntegerDriveMode)
{
  TempCalibFile file(R"({"1":{"homing_offset":0,"range_min":0,"range_max":4095,"drive_mode":1}})");
  const auto calibration = so101::Calibration::load(file.path(), {"shoulder"}, {{"shoulder", 1}});
  EXPECT_TRUE(calibration.at("shoulder").drive_mode);
}

}  // namespace
