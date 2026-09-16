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

#include <stdexcept>
#include <vector>

#include "feetech/bus.hpp"

namespace
{

std::vector<feetech::MotorConfig> arm_config(std::size_t count)
{
  std::vector<feetech::MotorConfig> motors;
  for (std::size_t i = 0; i < count; ++i) {
    feetech::MotorConfig cfg;
    cfg.id = static_cast<std::uint8_t>(i + 1);
    motors.push_back(cfg);
  }
  return motors;
}

TEST(BusLifecycle, OpenOnMissingPortFailsExplicitly)
{
  feetech::BusOptions options;
  options.port = "/dev/does-not-exist-feetech-test";
  options.motors = arm_config(2);
  options.simulated = false;

  feetech::Bus bus(options);
  EXPECT_FALSE(bus.open());
  EXPECT_EQ(bus.health().fault, feetech::Fault::PortOpenFailed);

  // Operations after a failed open fail without side effects.
  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::NotOpen);
}

TEST(BusLifecycle, OperationsBeforeOpenFail)
{
  feetech::BusOptions options;
  options.motors = arm_config(2);
  options.simulated = true;

  feetech::Bus bus(options);
  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::NotOpen);
  EXPECT_TRUE(samples.empty());
}

TEST(BusLifecycle, OperationsAfterCloseFail)
{
  feetech::BusOptions options;
  options.motors = arm_config(2);
  options.simulated = true;

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  bus.close();
  EXPECT_FALSE(bus.is_open());

  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::NotOpen);
}

TEST(BusLifecycle, OpenIsIdempotentAndCloseIsSafe)
{
  feetech::BusOptions options;
  options.motors = arm_config(1);
  options.simulated = true;

  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  EXPECT_TRUE(bus.open());
  bus.close();
  bus.close();  // idempotent
}

TEST(BusLifecycle, DuplicateMotorIdsRejectedAtConstruction)
{
  feetech::BusOptions options;
  options.motors = arm_config(2);
  options.motors[1].id = options.motors[0].id;
  options.simulated = true;
  EXPECT_THROW(feetech::Bus bus(options), std::invalid_argument);
}

TEST(BusLifecycle, EmptyMotorRegistryRejectedAtConstruction)
{
  feetech::BusOptions options;
  options.simulated = true;
  EXPECT_THROW(feetech::Bus bus(options), std::invalid_argument);
}

TEST(BusLifecycle, ReadTimeoutIncludesLerobotLatencyAndBaudrate)
{
  feetech::BusOptions options;
  options.motors = arm_config(6);
  options.simulated = true;
  feetech::Bus bus(options);
  EXPECT_EQ(bus.sync_read_timeout(1), std::chrono::milliseconds(51));
  EXPECT_EQ(bus.sync_read_timeout(6), std::chrono::milliseconds(52));
  EXPECT_THROW(bus.sync_read_timeout(0), std::invalid_argument);
  EXPECT_THROW(bus.sync_read_timeout(7), std::invalid_argument);

  options.baudrate = 9600;
  feetech::Bus slow_bus(options);
  EXPECT_EQ(slow_bus.sync_read_timeout(6), std::chrono::milliseconds(185));
  options.baudrate = 0;
  EXPECT_THROW(feetech::Bus invalid_bus(options), std::invalid_argument);
}

TEST(BusLifecycle, PingDistinguishesErrorResponseFromMissingMotor)
{
  feetech::BusOptions options;
  options.motors = arm_config(2);
  options.simulated = true;
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.ping_all().ok);

  bus.sim().set_response_status(2, 0x20);
  const auto started = std::chrono::steady_clock::now();
  const auto error = bus.ping_all(3, 500000);
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
  EXPECT_FALSE(error.ok);
  EXPECT_EQ(error.failed_id, 2U);
  EXPECT_EQ(error.fault, feetech::Fault::MotorMissing);
  EXPECT_NE(error.detail.find("answered with communication error state: 32"), std::string::npos);
  EXPECT_EQ(error.detail.find("cables"), std::string::npos);
  EXPECT_EQ(error.detail.find("not responding"), std::string::npos);

  bus.sim().set_responsive(2, false);
  const auto missing = bus.ping_all(2, 0);
  EXPECT_FALSE(missing.ok);
  EXPECT_EQ(missing.fault, feetech::Fault::MotorMissing);
  EXPECT_EQ(missing.failed_id, 2U);
  EXPECT_NE(missing.detail.find("not responding; check serial chain cables and power supply"),
    std::string::npos);
  bus.sim().clear_injections();
  EXPECT_TRUE(bus.ping_all().ok);
}

}  // namespace
