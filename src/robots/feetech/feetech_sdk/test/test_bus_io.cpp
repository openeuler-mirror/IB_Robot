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
#include "feetech/conversion.hpp"

namespace
{

feetech::BusOptions mobile_options()
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 6; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    cfg.name = "arm_joint_" + std::to_string(id);
    options.motors.push_back(cfg);
  }
  for (std::uint8_t id = 7; id <= 9; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    cfg.name = "wheel_" + std::to_string(id);
    cfg.mode = feetech::Mode::Wheel;
    options.motors.push_back(cfg);
  }
  return options;
}

TEST(BusIO, SyncReadReportsFreshStateInSiUnits)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(samples.size(), 9U);

  // Default simulated position is the center tick.
  EXPECT_NEAR(samples[0].position, 0.0, 1e-9);
  EXPECT_TRUE(samples[0].valid);
  EXPECT_EQ(bus.health().fault, feetech::Fault::None);
  EXPECT_TRUE(bus.health().ever_ok);
}

TEST(BusIO, SyncReadTransportFailureIsExplicit)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  std::vector<feetech::MotorSample> good;
  ASSERT_TRUE(bus.sync_read(good).ok);
  ASSERT_FALSE(good.empty());

  // Prime a second read with fresh data, then fail the transport.
  bus.sim().inject_sync_read_failure();
  std::vector<feetech::MotorSample> stale;
  const auto result = bus.sync_read(stale);

  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::SyncReadFailed);
  // The failed read MUST NOT present the previous values as fresh.
  EXPECT_TRUE(stale.empty());
  EXPECT_EQ(bus.health().consecutive_failures, 1U);
}

TEST(BusIO, MissingMotorIsIdentified)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  bus.sim().set_responsive(3, false);
  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);

  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::MotorMissing);
  EXPECT_EQ(result.failed_id, 3U);
  EXPECT_TRUE(samples.empty());
}

TEST(BusIO, ConsecutiveFailuresAccumulateAndResetOnSuccess)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  for (int i = 0; i < 3; ++i) {
    bus.sim().inject_sync_read_failure();
    std::vector<feetech::MotorSample> samples;
    EXPECT_FALSE(bus.sync_read(samples).ok);
  }
  EXPECT_EQ(bus.health().consecutive_failures, 3U);

  std::vector<feetech::MotorSample> samples;
  EXPECT_TRUE(bus.sync_read(samples).ok);
  EXPECT_EQ(bus.health().consecutive_failures, 0U);
  EXPECT_EQ(bus.health().fault, feetech::Fault::None);
}

TEST(BusIO, PositionWriteClampIsReported)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  std::vector<feetech::MotorTarget> targets;
  feetech::MotorTarget in_range;
  in_range.id = 1;
  in_range.position = 0.5;
  targets.push_back(in_range);
  feetech::MotorTarget out_of_range;
  out_of_range.id = 2;
  out_of_range.position = 100.0;
  targets.push_back(out_of_range);

  const auto result = bus.sync_write_positions(targets);
  EXPECT_TRUE(result.ok);
  ASSERT_EQ(result.clamped.size(), 1U);
  EXPECT_EQ(result.clamped[0], 2U);
}

TEST(BusIO, PositionWriteRejectsUnknownAndWrongMode)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  std::vector<feetech::MotorTarget> unknown;
  feetech::MotorTarget target;
  target.id = 42;
  target.position = 0.0;
  unknown.push_back(target);
  auto result = bus.sync_write_positions(unknown);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);
  EXPECT_EQ(result.failed_id, 42U);

  std::vector<feetech::MotorTarget> wrong_mode;
  target.id = 7;  // wheel motor
  wrong_mode.push_back(target);
  result = bus.sync_write_positions(wrong_mode);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);

  // And symmetrically: velocity writes reject position motors.
  std::vector<feetech::MotorTarget> velocity_to_arm;
  target.id = 1;
  velocity_to_arm.push_back(target);
  result = bus.sync_write_velocities(velocity_to_arm);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);
}

TEST(BusIO, PositionWriteRoundTripThroughSimulatedMotors)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  bus.sim().set_converge_step_ticks(4096);  // converge in a single read

  std::vector<feetech::MotorTarget> targets;
  feetech::MotorTarget target;
  target.id = 1;
  target.position = 0.5;
  targets.push_back(target);
  ASSERT_TRUE(bus.sync_write_positions(targets).ok);

  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  ASSERT_EQ(samples.size(), 9U);
  EXPECT_NEAR(samples[0].position, 0.5, 0.01);
}

TEST(BusIO, WheelVelocityCommandAccumulatesPosition)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  // Wheel mode is a configured property: apply configs before commanding.
  ASSERT_TRUE(bus.apply_all_configs().ok);
  bus.sim().set_position_ticks(7, 0);

  std::vector<feetech::MotorTarget> targets;
  feetech::MotorTarget target;
  target.id = 7;
  target.velocity = 1.0;  // rad/s
  targets.push_back(target);
  ASSERT_TRUE(bus.sync_write_velocities(targets).ok);

  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  // Each read advances the wheel by one step of the commanded speed.
  const double expected = feetech::steps_to_rad_per_s(feetech::rad_per_s_to_steps(1.0));
  EXPECT_NEAR(samples[6].position, expected, 1e-6);
  EXPECT_NEAR(samples[6].velocity, expected, 1e-6);
}

TEST(BusIO, ScopedReadIgnoresUnresponsiveOutOfSubsetMotor)
{
  // Shared-bus composition: an arm-scoped read must not fail because a wheel
  // motor (outside the subset) is unresponsive.
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  bus.sim().set_responsive(8, false);  // wheel motor drops off

  std::vector<feetech::MotorSample> arm_samples;
  const std::vector<std::uint8_t> arm_ids = {1, 2, 3, 4, 5, 6};
  const auto arm_result = bus.sync_read(arm_samples, arm_ids);
  EXPECT_TRUE(arm_result.ok);
  ASSERT_EQ(arm_samples.size(), 6U);
  EXPECT_EQ(arm_samples[0].id, 1U);

  // The full-bus read still fails and identifies the wheel.
  std::vector<feetech::MotorSample> all_samples;
  const auto all_result = bus.sync_read(all_samples);
  EXPECT_FALSE(all_result.ok);
  EXPECT_EQ(all_result.fault, feetech::Fault::MotorMissing);
  EXPECT_EQ(all_result.failed_id, 8U);
}

TEST(BusIO, ScopedConfigAndReleaseTouchOnlySubset)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  // Configure only the arm motors: wheels stay untouched (torque off,
  // default state).
  const std::vector<std::uint8_t> arm_ids = {1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(bus.apply_configs(arm_ids).ok);
  for (const std::uint8_t id : arm_ids) {
    EXPECT_TRUE(bus.sim().torque_enabled(id));
  }
  EXPECT_FALSE(bus.sim().torque_enabled(7));
  EXPECT_FALSE(bus.sim().torque_enabled(9));

  // Scoped release: only arm motors released, wheels (now configured) stay.
  const std::vector<std::uint8_t> wheel_ids = {7, 8, 9};
  ASSERT_TRUE(bus.apply_configs(wheel_ids).ok);
  ASSERT_TRUE(bus.emergency_release(arm_ids).ok);
  for (const std::uint8_t id : arm_ids) {
    EXPECT_FALSE(bus.sim().torque_enabled(id));
  }
  EXPECT_TRUE(bus.sim().torque_enabled(7));
}

TEST(BusIO, ScopedReadRejectsUnknownAndEmptyIds)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());

  std::vector<feetech::MotorSample> samples;
  auto result = bus.sync_read(samples, {42});
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);

  result = bus.sync_read(samples, {});
  EXPECT_FALSE(result.ok);
}

TEST(BusIO, RejectsNonFiniteAndDuplicateTargetsBeforeSending)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  feetech::MotorTarget target;
  target.id = 1;
  target.position = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(bus.sync_write_positions({target}).ok);
  target.position = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(bus.sync_write_positions({target}).ok);
  target.position = 1.0;
  EXPECT_FALSE(bus.sync_write_positions({target, target}).ok);
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  EXPECT_EQ(samples[0].position, 0.0);
}

TEST(BusIO, CalibrationCheckIsReadOnlyAndFailsOnUnprovisionedFirmware)
{
  auto options = mobile_options();
  options.motors.resize(1);
  options.motors[0].homing_offset = -42;
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  EXPECT_FALSE(bus.verify_calibration().ok);
  EXPECT_FALSE(bus.sim().torque_enabled(1));
  EXPECT_TRUE(bus.sim().eprom_locked(1));
  ASSERT_TRUE(bus.apply_all_configs().ok);
  ASSERT_TRUE(bus.emergency_release_all().ok);
  bus.sim().set_write_ack(1, false);
  EXPECT_TRUE(bus.verify_calibration().ok);
  EXPECT_FALSE(bus.sim().torque_enabled(1));
}

TEST(BusIO, CalibrationSkipsWheelMotorsOnMixedBus)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  ASSERT_TRUE(bus.emergency_release_all().ok);
  EXPECT_TRUE(bus.verify_calibration().ok);
  bus.sim().set_responsive(7, false);
  EXPECT_TRUE(bus.verify_calibration().ok);
  bus.sim().set_responsive(2, false);
  const auto result = bus.verify_calibration();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.failed_id, 2U);
}

TEST(BusIO, DuplicateAndOversizedReadsAreRejectedBeforeTransport)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  bus.sim().inject_sync_read_failure();
  std::vector<feetech::MotorSample> samples;
  for (const auto & ids : {std::vector<std::uint8_t>{1, 1}, std::vector<std::uint8_t>(250, 1)}) {
    const auto result = bus.sync_read(samples, ids);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.fault, feetech::Fault::WriteRejected);
    EXPECT_TRUE(samples.empty());
  }
  // Invalid calls did not consume the pending transport failure.
  EXPECT_EQ(bus.sync_read(samples).fault, feetech::Fault::SyncReadFailed);
  EXPECT_TRUE(bus.sync_read(samples).ok);
}

TEST(BusIO, VelocitySaturationSurvivesVendorEncodingAndIsReported)
{
  feetech::Bus bus(mobile_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  feetech::MotorTarget reverse;
  reverse.id = 7;
  reverse.velocity = -1e9;
  feetech::MotorTarget forward = reverse;
  forward.id = 8;
  forward.velocity = 1e9;
  feetech::MotorTarget normal = reverse;
  normal.id = 9;
  normal.velocity = 1.0;
  const auto result = bus.sync_write_velocities({reverse, forward, normal});
  ASSERT_TRUE(result.ok);
  EXPECT_EQ(result.clamped, (std::vector<std::uint8_t>{7, 8}));
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples, {7, 8, 9}).ok);
  EXPECT_DOUBLE_EQ(samples[0].velocity, feetech::steps_to_rad_per_s(-32767));
  EXPECT_DOUBLE_EQ(samples[1].velocity, feetech::steps_to_rad_per_s(32767));
}

TEST(BusIO, PingIsReadOnlyAndIdentifiesMissingMotor)
{
  feetech::Bus bus(mobile_options());
  EXPECT_EQ(bus.ping_all().fault, feetech::Fault::NotOpen);
  ASSERT_TRUE(bus.open());
  EXPECT_EQ(bus.ping_all(0).fault, feetech::Fault::WriteRejected);
  for (const auto & motor : bus.motors()) {
    bus.sim().set_write_ack(motor.id, false);
  }
  EXPECT_TRUE(bus.ping_all().ok);
  bus.sim().set_responsive(3, false);
  const auto started = std::chrono::steady_clock::now();
  const auto result = bus.ping_all(3, 1000);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.failed_id, 3U);
  EXPECT_EQ(result.fault, feetech::Fault::MotorMissing);
  EXPECT_NE(result.detail.find("cables"), std::string::npos);
  EXPECT_NE(result.detail.find("power"), std::string::npos);
  EXPECT_GE(elapsed, std::chrono::milliseconds(2));
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));
  for (const auto & motor : bus.motors()) {
    EXPECT_FALSE(bus.sim().torque_enabled(motor.id));
    EXPECT_TRUE(bus.sim().eprom_locked(motor.id));
  }
  bus.sim().set_responsive(3, true);
  EXPECT_TRUE(bus.ping_all().ok);
}

}  // namespace
