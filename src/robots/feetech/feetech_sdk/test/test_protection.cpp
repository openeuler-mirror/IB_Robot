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

#include <cstdint>
#include <string>
#include <vector>

#include "feetech/bus.hpp"

namespace
{

feetech::BusOptions options_for(std::vector<feetech::MotorConfig> motors)
{
  feetech::BusOptions options;
  options.simulated = true;
  options.motors = std::move(motors);
  return options;
}

feetech::MotorConfig plain_motor(std::uint8_t id)
{
  feetech::MotorConfig cfg;
  cfg.id = id;
  return cfg;
}

}  // namespace

TEST(Protection, DecodeNamesEveryFlag)
{
  EXPECT_EQ(feetech::decode_protection_bits(0x00), "none (0x00)");
  EXPECT_EQ(feetech::decode_protection_bits(feetech::kProtectionOverload), "OVERLOAD (0x20)");
  EXPECT_EQ(
    feetech::decode_protection_bits(
      feetech::kProtectionOverCurrent | feetech::kProtectionOverload),
    "OVER_CURRENT|OVERLOAD (0x28)");
  EXPECT_EQ(
    feetech::decode_protection_bits(
      feetech::kProtectionVoltage | feetech::kProtectionAngle | feetech::kProtectionOverheat),
    "VOLTAGE|ANGLE|OVERHEAT (0x07)");
  // Bit 4 is unused by the vendor table: the decode still reports the raw byte.
  EXPECT_EQ(feetech::decode_protection_bits(0x10), "none (0x10)");
}

TEST(Protection, OnlyTheAngleFaultInvalidatesFeedback)
{
  EXPECT_FALSE(feetech::protection_invalidates_feedback(0x00));
  EXPECT_FALSE(feetech::protection_invalidates_feedback(feetech::kProtectionOverload));
  EXPECT_FALSE(feetech::protection_invalidates_feedback(feetech::kProtectionOverCurrent));
  EXPECT_TRUE(feetech::protection_invalidates_feedback(feetech::kProtectionAngle));
  EXPECT_TRUE(feetech::protection_invalidates_feedback(
    feetech::kProtectionAngle | feetech::kProtectionOverload));
}

TEST(Protection, GroupReadKeepsStreamingWhenOneMotorReportsProtection)
{
  feetech::Bus bus(options_for({plain_motor(1), plain_motor(2), plain_motor(3)}));
  ASSERT_TRUE(bus.open());
  bus.sim().set_response_status(2, feetech::kProtectionOverload);

  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  ASSERT_TRUE(result.ok) << result.detail;
  ASSERT_EQ(samples.size(), 3U);
  // The healthy motors are untouched; only motor 2 carries the flag.
  EXPECT_EQ(samples[0].protection, 0U);
  EXPECT_EQ(samples[2].protection, 0U);
  EXPECT_EQ(samples[1].protection, feetech::kProtectionOverload);
  EXPECT_TRUE(samples[1].valid);
  EXPECT_EQ(bus.health().fault, feetech::Fault::None);

  // Clearing the firmware state returns the sample to healthy.
  bus.sim().set_response_status(2, 0);
  ASSERT_TRUE(bus.sync_read(samples).ok);
  EXPECT_EQ(samples[1].protection, 0U);
}

TEST(Protection, AngleSensorFaultMarksTheSampleUntrustworthy)
{
  feetech::Bus bus(options_for({plain_motor(1)}));
  ASSERT_TRUE(bus.open());
  bus.sim().set_response_status(1, feetech::kProtectionAngle);

  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  ASSERT_TRUE(result.ok) << result.detail;
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_EQ(samples[0].protection, feetech::kProtectionAngle);
  EXPECT_FALSE(samples[0].valid);
}
