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

#ifndef FEETECH__CONVERSION_HPP_
#define FEETECH__CONVERSION_HPP_

#include <cstdint>

namespace feetech
{

/// Encoder resolution: ticks per full revolution.
constexpr double kTicksPerRevolution = 4096.0;
/// Ticks per radian (STP/STS full circle = 4096 ticks).
constexpr double kTicksPerRad = kTicksPerRevolution / (2.0 * 3.14159265358979323846);
/// Raw tick value at the calibrated zero position.
constexpr int kCenterTick = 2048;
/// Hard tick range enforced on position commands.
constexpr int kMinTick = 0;
constexpr int kMaxTick = 4095;
/// Raw current register to amperes (STS3215 constant, proven on SO-101).
constexpr double kRawCurrentToAmpere = 0.0065;

/// Position-mode read: raw ticks (centered) -> radians.
double ticks_to_radians(int ticks);

/// Wheel-mode read: accumulated raw ticks -> radians.
double accumulated_ticks_to_radians(int ticks);

/// Position-mode write: radians -> raw ticks (centered, clamped to
/// [kMinTick, kMaxTick]).
int radians_to_ticks(double radians);

/// True when radians_to_ticks would clamp the value.
bool would_clamp_radians(double radians);

/// Velocity read: raw steps/s -> rad/s.
double steps_to_rad_per_s(int steps);

/// Velocity write: rad/s -> raw steps/s (sign-magnitude range [-32767, 32767]).
int rad_per_s_to_steps(double rad_per_s, bool * clamped = nullptr);

/// Current read: raw register value -> amperes.
double raw_current_to_ampere(int raw);

/// Homing offset write: signed ticks -> sign-magnitude word (sign bit 11).
std::uint16_t encode_homing_offset(int offset);

/// Feedback word decode: 15-bit sign-magnitude -> signed value.
int decode_sign_magnitude15(std::uint16_t word);

}  // namespace feetech

#endif  // FEETECH__CONVERSION_HPP_
