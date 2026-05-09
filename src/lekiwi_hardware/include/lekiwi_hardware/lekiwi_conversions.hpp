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

#ifndef LEKIWI_HARDWARE__LEKIWI_CONVERSIONS_HPP_
#define LEKIWI_HARDWARE__LEKIWI_CONVERSIONS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lekiwi_hardware
{

static constexpr double TICKS_PER_RAD = 4096.0 / (2.0 * M_PI);  // ~651.899

// Arm position read: raw ticks -> radians
inline double ticks_to_radians(int16_t raw_ticks)
{
  return (static_cast<double>(raw_ticks) - 2048.0) / TICKS_PER_RAD;
}

// Arm position write: radians -> ticks (clamped [0, 4095])
inline int16_t radians_to_ticks(double radians)
{
  double target = radians * TICKS_PER_RAD + 2048.0;
  target = std::clamp(target, 0.0, 4095.0);
  return static_cast<int16_t>(target);
}

// Speed read: raw steps/s -> rad/s
inline double steps_to_rad_s(int16_t raw_speed)
{
  return static_cast<double>(raw_speed) / TICKS_PER_RAD;
}

// Speed write: rad/s -> steps/s (clamped [-32768, 32767])
inline int16_t rad_s_to_steps(double rad_per_sec)
{
  double raw = rad_per_sec * TICKS_PER_RAD;
  raw = std::clamp(raw, -32768.0, 32767.0);
  return static_cast<int16_t>(raw);
}

// Decode 2-byte motor register with sign bit (bit 15)
inline int16_t decode_motor_register(uint8_t low_byte, uint8_t high_byte)
{
  int16_t val = (static_cast<int16_t>(high_byte) << 8) | low_byte;
  if (val & (1 << 15)) {
    val = -(val & 0x7FFF);
  }
  return val;
}

// Encode homing offset (bit 11 = sign for negative)
inline uint16_t encode_homing_offset(int offset)
{
  if (offset < 0) {
    return static_cast<uint16_t>(std::abs(offset)) | (1 << 11);
  }
  return static_cast<uint16_t>(offset);
}

}  // namespace lekiwi_hardware

#endif  // LEKIWI_HARDWARE__LEKIWI_CONVERSIONS_HPP_
