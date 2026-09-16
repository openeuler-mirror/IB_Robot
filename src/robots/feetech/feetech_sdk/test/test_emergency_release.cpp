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

#include "feetech/bus.hpp"

namespace
{

TEST(EmergencyRelease, ReleasesAllMotors)
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 4; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    options.motors.push_back(cfg);
  }

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  for (std::uint8_t id = 1; id <= 4; ++id) {
    ASSERT_TRUE(bus.sim().torque_enabled(id));
  }

  const auto result = bus.emergency_release_all();
  EXPECT_TRUE(result.ok);
  for (std::uint8_t id = 1; id <= 4; ++id) {
    EXPECT_FALSE(bus.sim().torque_enabled(id)) << "motor " << int(id);
  }
}

TEST(EmergencyRelease, ReportsUnreachableMotor)
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 3; ++id) {
    feetech::MotorConfig cfg;
    cfg.id = id;
    options.motors.push_back(cfg);
  }

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);

  // Motor 2 stops acknowledging every write: all three retries fail.
  bus.sim().set_write_ack(2, false);
  const auto result = bus.emergency_release_all();

  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::EmergencyPartiallyFailed);
  EXPECT_EQ(result.failed_id, 2U);
  EXPECT_NE(result.detail.find("2"), std::string::npos);
  // The other motors were still released.
  EXPECT_FALSE(bus.sim().torque_enabled(1));
  EXPECT_TRUE(bus.sim().torque_enabled(2));
  EXPECT_FALSE(bus.sim().torque_enabled(3));
}

TEST(EmergencyRelease, RetriesRecoverTransientAckFailures)
{
  feetech::BusOptions options;
  options.simulated = true;
  feetech::MotorConfig cfg;
  cfg.id = 1;
  options.motors.push_back(cfg);

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);

  // Two ack failures, then recovery: the three-attempt retry loop succeeds.
  bus.sim().fail_next_writes(1, 2);
  const auto result = bus.emergency_release_all();
  EXPECT_TRUE(result.ok);
  EXPECT_FALSE(bus.sim().torque_enabled(1));
}

}  // namespace
