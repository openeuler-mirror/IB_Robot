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

#include <cmath>

#include "feetech/conversion.hpp"

namespace
{

TEST(Conversion, CenterTickIsZeroRadians)
{
  EXPECT_DOUBLE_EQ(feetech::ticks_to_radians(feetech::kCenterTick), 0.0);
}

TEST(Conversion, TickRoundTrip)
{
  const int ticks = 2048 + 300;
  const double rad = feetech::ticks_to_radians(ticks);
  EXPECT_NEAR(feetech::radians_to_ticks(rad), ticks, 1.0);
}

TEST(Conversion, PositionClampsToTickRange)
{
  EXPECT_EQ(feetech::radians_to_ticks(100.0), feetech::kMaxTick);
  EXPECT_EQ(feetech::radians_to_ticks(-100.0), feetech::kMinTick);
  EXPECT_TRUE(feetech::would_clamp_radians(100.0));
  EXPECT_TRUE(feetech::would_clamp_radians(-100.0));
  EXPECT_FALSE(feetech::would_clamp_radians(0.5));
  EXPECT_FALSE(feetech::would_clamp_radians(-0.5));
}

TEST(Conversion, AccumulatedTicksAreNotCentered)
{
  // Wheel mode: 4096 ticks == one full revolution == 2*pi radians.
  EXPECT_NEAR(feetech::accumulated_ticks_to_radians(4096), 2.0 * M_PI, 1e-9);
  EXPECT_DOUBLE_EQ(feetech::accumulated_ticks_to_radians(0), 0.0);
}

TEST(Conversion, VelocityRoundTrip)
{
  const double rad_s = 1.5;
  const int steps = feetech::rad_per_s_to_steps(rad_s);
  // Tick quantisation bounds the round-trip error to one step.
  EXPECT_NEAR(feetech::steps_to_rad_per_s(steps), rad_s, 2.0 / feetech::kTicksPerRad);
}

TEST(Conversion, VelocityClampsToSignMagnitude)
{
  EXPECT_EQ(feetech::rad_per_s_to_steps(1e6), 32767);
  bool clamped = false;
  EXPECT_EQ(feetech::rad_per_s_to_steps(-1e6, &clamped), -32767);
  EXPECT_TRUE(clamped);
  EXPECT_EQ(feetech::rad_per_s_to_steps(0, &clamped), 0);
  EXPECT_FALSE(clamped);
}

TEST(Conversion, CurrentUsesProvenScale)
{
  EXPECT_NEAR(feetech::raw_current_to_ampere(1000), 6.5, 1e-9);
  EXPECT_DOUBLE_EQ(feetech::raw_current_to_ampere(0), 0.0);
}

TEST(Conversion, HomingOffsetSignMagnitude)
{
  // Positive offsets pass through; negative offsets set sign bit 11.
  EXPECT_EQ(feetech::encode_homing_offset(0), 0x0000U);
  EXPECT_EQ(feetech::encode_homing_offset(123), 123U);
  EXPECT_EQ(feetech::encode_homing_offset(-123), 123U | (1U << 11));
}

TEST(Conversion, SignMagnitudeDecode)
{
  // Sign-magnitude with bit 15 as the sign flag (vendor SyncWriteSpe
  // encoding; matches the proven SO-101/LeKiwi decode paths).
  EXPECT_EQ(feetech::decode_sign_magnitude15(0x0000U), 0);
  EXPECT_EQ(feetech::decode_sign_magnitude15(0x0064U), 100);
  EXPECT_EQ(feetech::decode_sign_magnitude15(0x8064U), -100);
  EXPECT_EQ(feetech::decode_sign_magnitude15(0x7FFFU), 32767);
  EXPECT_EQ(feetech::decode_sign_magnitude15(0xFFFFU), -32767);
}

}  // namespace
