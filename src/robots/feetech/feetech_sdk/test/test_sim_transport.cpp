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

// Tests for the simulated transport itself (the spec's "Simulated transport
// for tests" requirement): convergence behavior and failure injection, driven
// through the real Bus + real vendor protocol stack.

#include <gtest/gtest.h>

#include <vector>

#include "feetech/bus.hpp"
#include "feetech/conversion.hpp"

namespace
{

TEST(SimTransport, PositionsConvergeTowardCommandedTargets)
{
  feetech::BusOptions options;
  options.simulated = true;
  feetech::MotorConfig cfg;
  cfg.id = 1;
  options.motors.push_back(cfg);

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  bus.sim().set_converge_step_ticks(100);
  bus.sim().set_position_ticks(1, 2048);

  std::vector<feetech::MotorTarget> targets;
  feetech::MotorTarget target;
  target.id = 1;
  target.position = 1.0;  // ~2698 ticks
  targets.push_back(target);
  ASSERT_TRUE(bus.sync_write_positions(targets).ok);

  double previous = 0.0;
  bool moved = false;
  for (int read = 0; read < 10; ++read) {
    std::vector<feetech::MotorSample> samples;
    ASSERT_TRUE(bus.sync_read(samples).ok);
    EXPECT_GE(samples[0].position, previous);  // never moves backwards
    if (samples[0].position > previous + 1e-9) {
      moved = true;
    }
    previous = samples[0].position;
  }
  EXPECT_TRUE(moved);
  EXPECT_NEAR(previous, 1.0, 0.02);
}

TEST(SimTransport, SeedPositionIsObservable)
{
  feetech::BusOptions options;
  options.simulated = true;
  feetech::MotorConfig cfg;
  cfg.id = 1;
  options.motors.push_back(cfg);

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  bus.sim().set_position_ticks(1, 3048);

  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  EXPECT_NEAR(samples[0].position, feetech::ticks_to_radians(3048), 1e-9);
}

TEST(SimTransport, TorqueOffPreventsMotionAndReactivationDiscardsStaleGoal)
{
  feetech::BusOptions options;
  options.simulated = true;
  feetech::MotorConfig cfg;
  cfg.id = 1;
  options.motors.push_back(cfg);
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  feetech::MotorTarget target;
  target.id = 1;
  target.position = 1.0;
  ASSERT_TRUE(bus.sync_write_positions({target}).ok);
  ASSERT_TRUE(bus.emergency_release_all().ok);
  std::vector<feetech::MotorSample> samples;
  ASSERT_TRUE(bus.sync_read(samples).ok);
  EXPECT_EQ(bus.sim().position_ticks(1), 2048);
  EXPECT_EQ(samples[0].velocity, 0.0);
  ASSERT_TRUE(bus.apply_all_configs().ok);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(bus.sync_read(samples).ok);
    EXPECT_EQ(bus.sim().position_ticks(1), 2048);
  }
}

TEST(SimTransport, FailureInjectionMatchesRealFailureSemantics)
{
  // Two motors: when one stops answering, the group read still receives the
  // other motor's packets, so the failure classifies as MotorMissing rather
  // than a transport-level SyncReadFailed.
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 2; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    options.motors.push_back(cfg);
  }

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  bus.sim().inject_sync_read_failure();
  std::vector<feetech::MotorSample> samples;
  const auto transport_failure = bus.sync_read(samples);
  ASSERT_FALSE(transport_failure.ok);
  EXPECT_EQ(transport_failure.fault, feetech::Fault::SyncReadFailed);

  bus.sim().set_responsive(1, false);
  const auto missing = bus.sync_read(samples);
  ASSERT_FALSE(missing.ok);
  EXPECT_EQ(missing.fault, feetech::Fault::MotorMissing);
  EXPECT_EQ(missing.failed_id, 1U);

  // Clearing injections restores healthy operation.
  bus.sim().clear_injections();
  EXPECT_TRUE(bus.sync_read(samples).ok);
}

TEST(SimTransport, ResponseStatusReachesReadAndWriteFailureChecks)
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 2; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    options.motors.push_back(cfg);
  }
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  bus.sim().set_response_status(2, 0x20);
  std::vector<feetech::MotorSample> samples;
  const auto feedback = bus.sync_read(samples);
  EXPECT_FALSE(feedback.ok);
  EXPECT_EQ(feedback.fault, feetech::Fault::SyncReadFailed);
  EXPECT_EQ(feedback.failed_id, 2U);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(bus.health().fault, feetech::Fault::SyncReadFailed);
  EXPECT_FALSE(bus.verify_calibration().ok);
  const auto release = bus.emergency_release_all();
  EXPECT_FALSE(release.ok);
  EXPECT_EQ(release.fault, feetech::Fault::EmergencyPartiallyFailed);
  EXPECT_EQ(release.failed_id, 2U);
  bus.sim().clear_injections();
  EXPECT_TRUE(bus.sync_read(samples).ok);
  EXPECT_TRUE(bus.verify_calibration().ok);
  EXPECT_TRUE(bus.emergency_release_all().ok);
}

}  // namespace
