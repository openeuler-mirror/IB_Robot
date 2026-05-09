// Copyright 2025 yanhan
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
#include <cstdint>

#include "lekiwi_hardware/lekiwi_conversions.hpp"

using lekiwi_hardware::TICKS_PER_RAD;
using lekiwi_hardware::ticks_to_radians;
using lekiwi_hardware::radians_to_ticks;
using lekiwi_hardware::steps_to_rad_s;
using lekiwi_hardware::rad_s_to_steps;
using lekiwi_hardware::decode_motor_register;
using lekiwi_hardware::encode_homing_offset;

// ============================================================================
// 1. TestTicksToRadians
// ============================================================================
TEST(TestTicksToRadians, CenterPosition)
{
  EXPECT_DOUBLE_EQ(ticks_to_radians(2048), 0.0);
}

TEST(TestTicksToRadians, MinPosition)
{
  double expected = -2048.0 / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(0), expected, 1e-12);
}

TEST(TestTicksToRadians, MaxPosition)
{
  double expected = (4095.0 - 2048.0) / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(4095), expected, 1e-12);
}

TEST(TestTicksToRadians, QuarterTurnPositive)
{
  double expected = (3072.0 - 2048.0) / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(3072), expected, 1e-12);
}

TEST(TestTicksToRadians, QuarterTurnNegative)
{
  double expected = (1024.0 - 2048.0) / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(1024), expected, 1e-12);
}

TEST(TestTicksToRadians, SmallPositive)
{
  double expected = 1.0 / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(2049), expected, 1e-12);
}

TEST(TestTicksToRadians, SmallNegative)
{
  double expected = -1.0 / TICKS_PER_RAD;
  EXPECT_NEAR(ticks_to_radians(2047), expected, 1e-12);
}

TEST(TestTicksToRadians, Symmetry)
{
  int16_t offset = 500;
  double pos_val = ticks_to_radians(2048 + offset);
  double neg_val = ticks_to_radians(2048 - offset);
  EXPECT_NEAR(pos_val, -neg_val, 1e-12);
}

// ============================================================================
// 2. TestRadiansToTicks
// ============================================================================
TEST(TestRadiansToTicks, ZeroRadians)
{
  EXPECT_EQ(radians_to_ticks(0.0), 2048);
}

TEST(TestRadiansToTicks, PositivePi)
{
  // M_PI maps to ~4096, clamped to 4095
  EXPECT_EQ(radians_to_ticks(M_PI), 4095);
}

TEST(TestRadiansToTicks, NegativePi)
{
  // -M_PI maps to ~0, clamped to 0
  EXPECT_EQ(radians_to_ticks(-M_PI), 0);
}

TEST(TestRadiansToTicks, PositiveHalfPi)
{
  double expected = M_PI / 2.0 * TICKS_PER_RAD + 2048.0;
  EXPECT_NEAR(radians_to_ticks(M_PI / 2.0), static_cast<int16_t>(expected), 1);
}

TEST(TestRadiansToTicks, NegativeHalfPi)
{
  double expected = -M_PI / 2.0 * TICKS_PER_RAD + 2048.0;
  EXPECT_NEAR(radians_to_ticks(-M_PI / 2.0), static_cast<int16_t>(expected), 1);
}

TEST(TestRadiansToTicks, ClampAboveMax)
{
  EXPECT_EQ(radians_to_ticks(100.0), 4095);
}

TEST(TestRadiansToTicks, ClampBelowMin)
{
  EXPECT_EQ(radians_to_ticks(-100.0), 0);
}

TEST(TestRadiansToTicks, JustBelowMax)
{
  double rad = (4094.5 - 2048.0) / TICKS_PER_RAD;
  int16_t ticks = radians_to_ticks(rad);
  EXPECT_TRUE(ticks == 4094 || ticks == 4095);
}

TEST(TestRadiansToTicks, JustAboveMin)
{
  double rad = (0.5 - 2048.0) / TICKS_PER_RAD;
  int16_t ticks = radians_to_ticks(rad);
  EXPECT_TRUE(ticks == 0 || ticks == 1);
}

// ============================================================================
// 3. TestStepsToRadS
// ============================================================================
TEST(TestStepsToRadS, ZeroSpeed)
{
  EXPECT_DOUBLE_EQ(steps_to_rad_s(0), 0.0);
}

TEST(TestStepsToRadS, PositiveSpeed)
{
  EXPECT_NEAR(steps_to_rad_s(652), 652.0 / TICKS_PER_RAD, 1e-12);
}

TEST(TestStepsToRadS, NegativeSpeed)
{
  EXPECT_NEAR(steps_to_rad_s(-652), -652.0 / TICKS_PER_RAD, 1e-12);
}

TEST(TestStepsToRadS, MaxSpeed)
{
  EXPECT_NEAR(steps_to_rad_s(32767), 32767.0 / TICKS_PER_RAD, 1e-9);
}

TEST(TestStepsToRadS, MinSpeed)
{
  EXPECT_NEAR(steps_to_rad_s(-32768), -32768.0 / TICKS_PER_RAD, 1e-9);
}

TEST(TestStepsToRadS, UnityTick)
{
  EXPECT_NEAR(steps_to_rad_s(1), 1.0 / TICKS_PER_RAD, 1e-12);
}

// ============================================================================
// 4. TestRadSToSteps
// ============================================================================
TEST(TestRadSToSteps, ZeroSpeed)
{
  EXPECT_EQ(rad_s_to_steps(0.0), 0);
}

TEST(TestRadSToSteps, OneRadPerSec)
{
  EXPECT_NEAR(rad_s_to_steps(1.0), static_cast<int16_t>(1.0 * TICKS_PER_RAD), 1);
}

TEST(TestRadSToSteps, NegativeOneRadPerSec)
{
  EXPECT_NEAR(rad_s_to_steps(-1.0), static_cast<int16_t>(-1.0 * TICKS_PER_RAD), 1);
}

TEST(TestRadSToSteps, ClampAboveMax)
{
  EXPECT_EQ(rad_s_to_steps(100.0), 32767);
}

TEST(TestRadSToSteps, ClampBelowMin)
{
  EXPECT_EQ(rad_s_to_steps(-100.0), -32768);
}

TEST(TestRadSToSteps, MaxUnclamped)
{
  double rad = 32767.0 / TICKS_PER_RAD;
  // Floating-point roundtrip may lose 1 step
  int16_t result = rad_s_to_steps(rad);
  EXPECT_TRUE(result == 32767 || result == 32766);
}

TEST(TestRadSToSteps, MinUnclamped)
{
  double rad = -32768.0 / TICKS_PER_RAD;
  EXPECT_EQ(rad_s_to_steps(rad), -32768);
}

TEST(TestRadSToSteps, JustAboveBoundary)
{
  double rad = 32768.0 / TICKS_PER_RAD;
  EXPECT_EQ(rad_s_to_steps(rad), 32767);
}

// ============================================================================
// 5. TestDecodeMotorRegister
// ============================================================================
TEST(TestDecodeMotorRegister, Zero)
{
  EXPECT_EQ(decode_motor_register(0x00, 0x00), 0);
}

TEST(TestDecodeMotorRegister, PositiveOne)
{
  EXPECT_EQ(decode_motor_register(0x01, 0x00), 1);
}

TEST(TestDecodeMotorRegister, NegativeOne)
{
  EXPECT_EQ(decode_motor_register(0x01, 0x80), -1);
}

TEST(TestDecodeMotorRegister, PositiveSmall)
{
  EXPECT_EQ(decode_motor_register(0xFF, 0x00), 255);
}

TEST(TestDecodeMotorRegister, NegativeSmall)
{
  EXPECT_EQ(decode_motor_register(0xFF, 0x80), -255);
}

TEST(TestDecodeMotorRegister, PositiveMax)
{
  EXPECT_EQ(decode_motor_register(0xFF, 0x7F), 32767);
}

TEST(TestDecodeMotorRegister, NegativeMaxMagnitude)
{
  EXPECT_EQ(decode_motor_register(0xFF, 0xFF), -32767);
}

TEST(TestDecodeMotorRegister, CenterTick)
{
  EXPECT_EQ(decode_motor_register(0x00, 0x08), 2048);
}

TEST(TestDecodeMotorRegister, JustAboveCenter)
{
  EXPECT_EQ(decode_motor_register(0x01, 0x08), 2049);
}

TEST(TestDecodeMotorRegister, JustBelowCenter)
{
  EXPECT_EQ(decode_motor_register(0xFF, 0x07), 2047);
}

TEST(TestDecodeMotorRegister, ZeroMagnitudeWithSignBit)
{
  EXPECT_EQ(decode_motor_register(0x00, 0x80), 0);
}

// ============================================================================
// 6. TestEncodeHomingOffset
// ============================================================================
TEST(TestEncodeHomingOffset, ZeroOffset)
{
  EXPECT_EQ(encode_homing_offset(0), 0x0000u);
}

TEST(TestEncodeHomingOffset, PositiveOne)
{
  EXPECT_EQ(encode_homing_offset(1), 0x0001u);
}

TEST(TestEncodeHomingOffset, NegativeOne)
{
  EXPECT_EQ(encode_homing_offset(-1), 0x0801u);
}

TEST(TestEncodeHomingOffset, PositiveLarge)
{
  EXPECT_EQ(encode_homing_offset(100), 100u);
}

TEST(TestEncodeHomingOffset, NegativeLarge)
{
  EXPECT_EQ(encode_homing_offset(-100), (100u | 0x800u));
}

TEST(TestEncodeHomingOffset, MaxPositive)
{
  EXPECT_EQ(encode_homing_offset(2047), 2047u);
}

TEST(TestEncodeHomingOffset, MaxNegative)
{
  EXPECT_EQ(encode_homing_offset(-2047), (2047u | 0x800u));
}

TEST(TestEncodeHomingOffset, Positive50)
{
  EXPECT_EQ(encode_homing_offset(50), 50u);
}

TEST(TestEncodeHomingOffset, Negative50)
{
  EXPECT_EQ(encode_homing_offset(-50), (50u | 0x800u));
}

// ============================================================================
// 7. TestRoundtrip
// ============================================================================
TEST(TestRoundtrip, TicksRadiansRoundtrip)
{
  for (int t = 0; t <= 4095; t += 204) {
    int16_t original = static_cast<int16_t>(t);
    double rad = ticks_to_radians(original);
    int16_t recovered = radians_to_ticks(rad);
    EXPECT_LE(std::abs(recovered - original), 1)
      << "original=" << original << " recovered=" << recovered;
  }
}

TEST(TestRoundtrip, StepsRoundtrip)
{
  for (int s = -32767; s <= 32767; s += 3276) {
    int16_t original = static_cast<int16_t>(s);
    double rad = steps_to_rad_s(original);
    int16_t recovered = rad_s_to_steps(rad);
    // Allow ±1 due to floating-point truncation in roundtrip
    EXPECT_NEAR(recovered, original, 1)
      << "original=" << original << " recovered=" << recovered;
  }
}

TEST(TestRoundtrip, RadiansRoundtrip)
{
  double step = M_PI / 10.0;
  for (double r = -M_PI; r <= M_PI; r += step) {
    int16_t ticks = radians_to_ticks(r);
    double recovered = ticks_to_radians(ticks);
    // Allow up to 2 ticks of quantization error (clamping at boundary adds 1 tick)
    EXPECT_LE(std::abs(recovered - r), 2.0 / TICKS_PER_RAD)
      << "r=" << r << " recovered=" << recovered << " ticks=" << ticks;
  }
}

TEST(TestRoundtrip, RegisterDecodeConsistency)
{
  for (int v = -32767; v <= 32767; v += 3276) {
    int16_t original = static_cast<int16_t>(v);
    // Encode: positive values stored directly, negative uses sign bit
    uint8_t low, high;
    if (original >= 0) {
      low = static_cast<uint8_t>(original & 0xFF);
      high = static_cast<uint8_t>((original >> 8) & 0xFF);
    } else {
      int16_t encoded = (std::abs(static_cast<int>(original))) | (1 << 15);
      low = static_cast<uint8_t>(encoded & 0xFF);
      high = static_cast<uint8_t>((encoded >> 8) & 0xFF);
    }
    int16_t decoded = decode_motor_register(low, high);
    EXPECT_EQ(decoded, original)
      << "original=" << original << " decoded=" << decoded;
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
