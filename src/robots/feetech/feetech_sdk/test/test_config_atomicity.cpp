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

#include <vector>
#include <limits>

#include "feetech/bus.hpp"

namespace
{

feetech::BusOptions three_arm_options()
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 3; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    options.motors.push_back(cfg);
  }
  return options;
}

TEST(ConfigAtomicity, SuccessfulApplyLeavesMotorsEnabledAndLocked)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());

  const auto result = bus.apply_all_configs();
  ASSERT_TRUE(result.ok);
  for (std::uint8_t id = 1; id <= 3; ++id) {
    EXPECT_TRUE(bus.sim().torque_enabled(id)) << "motor " << int(id);
    EXPECT_TRUE(bus.sim().eprom_locked(id)) << "motor " << int(id);
  }
}

TEST(ConfigAtomicity, MidSequenceFailureRollsBackTouchedMotors)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());

  // Motor 2 rejects writes to the homing-offset register: after motor 1 was
  // fully configured and motor 2 passed "unlock EPROM", the "write homing
  // offset" step fails with motor 2's EPROM left unlocked.
  bus.sim().reject_register_write(2, 31);

  const auto result = bus.apply_all_configs();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::ConfigFailed);
  EXPECT_EQ(result.failed_id, 2U);
  EXPECT_NE(result.detail.find("write homing offset"), std::string::npos)
    << "detail was: " << result.detail;

  // Every touched motor is left torque-disabled with EPROM locked (the
  // unlocked motor is relocked by the rollback).
  for (std::uint8_t id = 1; id <= 2; ++id) {
    EXPECT_FALSE(bus.sim().torque_enabled(id)) << "motor " << int(id);
    EXPECT_TRUE(bus.sim().eprom_locked(id)) << "motor " << int(id);
  }
  // Motor 3 was never touched: torque still disabled (default), EPROM locked.
  EXPECT_FALSE(bus.sim().torque_enabled(3));
  EXPECT_TRUE(bus.sim().eprom_locked(3));
}

TEST(ConfigAtomicity, LaterStepFailureStillRollsBackEverything)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());

  // Motor 3 rejects only torque-enable writes (value 1): the final
  // "enable torque" step fails after EPROM was already relocked.
  bus.sim().reject_register_write(3, 40, 1);

  const auto result = bus.apply_all_configs();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::ConfigFailed);
  EXPECT_EQ(result.failed_id, 3U);
  EXPECT_NE(result.detail.find("enable torque"), std::string::npos)
    << "detail was: " << result.detail;

  // Motors 1 and 2 were fully configured before the failure: rolled back.
  EXPECT_FALSE(bus.sim().torque_enabled(1));
  EXPECT_FALSE(bus.sim().torque_enabled(2));
  EXPECT_FALSE(bus.sim().torque_enabled(3));
  for (std::uint8_t id = 1; id <= 3; ++id) {
    EXPECT_TRUE(bus.sim().eprom_locked(id)) << "motor " << int(id);
  }
}

TEST(ConfigAtomicity, SingleMotorApplyIsAtomic)
{
  feetech::BusOptions options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  feetech::MotorConfig cfg = options.motors[0];
  cfg.homing_offset = -50;
  cfg.range_min = 100;
  cfg.range_max = 4000;

  bus.sim().fail_next_writes(1, 3);  // torque-off ok, then unlock fails? No:
  // fail_next_writes(1, 3) makes the first THREE ack'd writes fail, i.e.
  // "disable torque", "unlock EPROM", ... Use it to fail a bounded number of
  // writes and verify recovery is possible afterwards.
  const auto result = bus.apply_motor_config(cfg);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::ConfigFailed);
  EXPECT_FALSE(bus.sim().torque_enabled(1));
  EXPECT_TRUE(bus.sim().eprom_locked(1));

  // After clearing injections the same config applies cleanly.
  bus.sim().clear_injections();
  const auto retry = bus.apply_motor_config(cfg);
  EXPECT_TRUE(retry.ok) << retry.detail;
  EXPECT_TRUE(bus.sim().torque_enabled(1));
  EXPECT_TRUE(bus.sim().eprom_locked(1));
}

TEST(ConfigAtomicity, UnregisteredMotorIsRejected)
{
  feetech::BusOptions options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  feetech::MotorConfig foreign;
  foreign.id = 9;
  const auto result = bus.apply_motor_config(foreign);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);
}

TEST(ConfigAtomicity, RejectedHoldSeedNeverEnablesTorque)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());
  bus.sim().reject_register_write(2, 41);
  const auto result = bus.apply_all_configs();
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.detail.find("seed activation hold"), std::string::npos);
  for (std::uint8_t id = 1; id <= 3; ++id) {
    EXPECT_FALSE(bus.sim().torque_enabled(id));
    EXPECT_TRUE(bus.sim().eprom_locked(id));
  }
}

TEST(ConfigAtomicity, FailedTorqueRollbackIsAnEmergency)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  bus.sim().reject_register_write(2, 40, 0);
  const auto result = bus.apply_all_configs();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::EmergencyPartiallyFailed);
  EXPECT_EQ(result.failed_id, 2U);
  EXPECT_NE(result.detail.find("ROLLBACK INCOMPLETE"), std::string::npos);
  EXPECT_TRUE(bus.sim().torque_enabled(2));
  EXPECT_FALSE(bus.sim().torque_enabled(1));
}

TEST(ConfigAtomicity, FailedRelockIsReportedSeparately)
{
  feetech::Bus bus(three_arm_options());
  ASSERT_TRUE(bus.open());
  bus.sim().reject_register_write(2, 31);
  bus.sim().reject_register_write(2, 55, 1);
  const auto result = bus.apply_all_configs();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::ConfigFailed);
  EXPECT_NE(result.detail.find("EPROM left unlocked on motors: 2"), std::string::npos);
  EXPECT_FALSE(bus.sim().torque_enabled(2));
  EXPECT_FALSE(bus.sim().eprom_locked(2));
}

TEST(ConfigAtomicity, InvalidRuntimeConfigCannotWriteRegisters)
{
  const auto options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  std::vector<feetech::MotorConfig> invalid;
  for (const int offset : {-2048, 2048, std::numeric_limits<int>::min()}) {
    auto cfg = options.motors[0];
    cfg.homing_offset = offset;
    invalid.push_back(cfg);
  }
  auto cfg = options.motors[0];
  cfg.range_min = -1;
  invalid.push_back(cfg);
  cfg = options.motors[0];
  cfg.range_max = 100000;
  invalid.push_back(cfg);
  cfg.range_max = -1;
  invalid.push_back(cfg);
  cfg.range_min = 4096;
  cfg.range_max = 4095;
  invalid.push_back(cfg);
  // A write would consume this injection, even if firmware remained unchanged.
  bus.sim().fail_next_writes(1, 1);
  for (const auto & candidate : invalid) {
    const auto result = bus.apply_motor_config(candidate);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);
    EXPECT_TRUE(bus.sim().torque_enabled(1));
    EXPECT_TRUE(bus.sim().eprom_locked(1));
    EXPECT_TRUE(bus.verify_calibration().ok);
    auto bad_options = options;
    bad_options.motors[0] = candidate;
    EXPECT_THROW(feetech::Bus rejected(bad_options), std::invalid_argument);
  }
  EXPECT_FALSE(bus.apply_motor_config(options.motors[0]).ok);
}

TEST(ConfigAtomicity, SuccessfulModeChangeUpdatesCommandGuardsAndRegistry)
{
  const auto options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  auto cfg = options.motors[0];
  cfg.mode = feetech::Mode::Wheel;
  ASSERT_TRUE(bus.apply_motor_config(cfg).ok);
  EXPECT_EQ(bus.motors()[0].mode, feetech::Mode::Wheel);
  feetech::MotorTarget target;
  target.id = 1;
  target.position = 0.5;
  target.velocity = 0.5;
  EXPECT_EQ(bus.sync_write_positions({target}).fault, feetech::Fault::WriteRejected);
  EXPECT_TRUE(bus.sync_write_velocities({target}).ok);
  bus.sim().set_responsive(1, false);
  EXPECT_TRUE(bus.verify_calibration().ok);
  bus.sim().set_responsive(1, true);
  // apply_all_configs must retain the updated wheel configuration.
  ASSERT_TRUE(bus.apply_all_configs().ok);
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_EQ(samples.back().id, 1U);
  cfg.mode = feetech::Mode::Position;
  cfg.homing_offset = -42;
  ASSERT_TRUE(bus.apply_motor_config(cfg).ok);
  EXPECT_TRUE(bus.verify_calibration().ok);
  EXPECT_TRUE(bus.sync_write_positions({target}).ok);
  EXPECT_EQ(bus.sync_write_velocities({target}).fault, feetech::Fault::WriteRejected);
  cfg.mode = feetech::Mode::Wheel;
  bus.sim().reject_register_write(1, 31);
  EXPECT_FALSE(bus.apply_motor_config(cfg).ok);
  EXPECT_EQ(bus.motors()[0].mode, feetech::Mode::Position);
}

TEST(ConfigAtomicity, FailedConfigAfterModeWriteReconcilesOnlyMode)
{
  const auto options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  auto cfg = options.motors[0];
  cfg.mode = feetech::Mode::Wheel;
  cfg.homing_offset = -42;
  cfg.kp = 25;
  cfg.profile_speed = 100;
  // Reject the last step, after MODE was written and EPROM was relocked.
  bus.sim().reject_register_write(1, 40, 1);
  const auto result = bus.apply_motor_config(cfg);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::ConfigFailed);
  EXPECT_NE(result.detail.find("enable torque"), std::string::npos);
  EXPECT_TRUE(bus.sim().eprom_locked(1));
  EXPECT_FALSE(bus.sim().torque_enabled(1));
  EXPECT_EQ(bus.motors()[0].mode, feetech::Mode::Wheel);
  EXPECT_EQ(bus.motors()[0].homing_offset, options.motors[0].homing_offset);
  EXPECT_EQ(bus.motors()[0].kp, options.motors[0].kp);
  EXPECT_EQ(bus.motors()[0].profile_speed, options.motors[0].profile_speed);
  feetech::MotorTarget target;
  target.id = 1;
  EXPECT_EQ(bus.sync_write_positions({target}).fault, feetech::Fault::WriteRejected);
  EXPECT_TRUE(bus.sync_write_velocities({target}).ok);
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_EQ(samples.back().id, 1U);  // moved to the wheel group
  bus.sim().clear_injections();
  ASSERT_TRUE(bus.apply_all_configs().ok);
  EXPECT_EQ(bus.sync_write_positions({target}).fault, feetech::Fault::WriteRejected);
}

TEST(ConfigAtomicity, FailedModeReadbackKeepsRegistryUnchanged)
{
  const auto options = three_arm_options();
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  auto cfg = options.motors[0];
  cfg.mode = feetech::Mode::Wheel;
  // Writes still work, but neither the failure-path readback nor other reads answer.
  bus.sim().set_responsive(1, false);
  bus.sim().reject_register_write(1, 40, 1);
  const auto result = bus.apply_motor_config(cfg);
  ASSERT_FALSE(result.ok);
  EXPECT_NE(result.detail.find("enable torque"), std::string::npos);
  EXPECT_EQ(bus.motors()[0].mode, feetech::Mode::Position);
  feetech::MotorTarget target;
  target.id = 1;
  EXPECT_EQ(bus.sync_write_velocities({target}).fault, feetech::Fault::WriteRejected);
  bus.sim().set_responsive(1, true);
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  EXPECT_EQ(samples.front().id, 1U);
}

}  // namespace
