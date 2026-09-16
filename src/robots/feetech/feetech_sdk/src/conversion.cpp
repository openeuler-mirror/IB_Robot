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

#include "feetech/conversion.hpp"

#include <algorithm>
#include <cmath>

namespace feetech
{

double ticks_to_radians(int ticks)
{
  return (static_cast<double>(ticks) - static_cast<double>(kCenterTick)) / kTicksPerRad;
}

double accumulated_ticks_to_radians(int ticks)
{
  return static_cast<double>(ticks) / kTicksPerRad;
}

int radians_to_ticks(double radians)
{
  double target = radians * kTicksPerRad + static_cast<double>(kCenterTick);
  target = std::clamp(target, static_cast<double>(kMinTick), static_cast<double>(kMaxTick));
  return static_cast<int>(target);
}

bool would_clamp_radians(double radians)
{
  const double target = radians * kTicksPerRad + static_cast<double>(kCenterTick);
  return target < static_cast<double>(kMinTick) || target > static_cast<double>(kMaxTick);
}

double steps_to_rad_per_s(int steps)
{
  return static_cast<double>(steps) / kTicksPerRad;
}

int rad_per_s_to_steps(double rad_per_s, bool * clamped)
{
  const double raw = rad_per_s * kTicksPerRad;
  // Vendor velocity words are sign-magnitude, so -32768 would encode zero.
  const double bounded = std::clamp(raw, -32767.0, 32767.0);
  if (clamped != nullptr) {
    *clamped = bounded != raw;
  }
  return static_cast<int>(bounded);
}

double raw_current_to_ampere(int raw)
{
  return static_cast<double>(raw) * kRawCurrentToAmpere;
}

std::uint16_t encode_homing_offset(int offset)
{
  if (offset < 0) {
    return static_cast<std::uint16_t>(std::abs(offset)) | (1U << 11);
  }
  return static_cast<std::uint16_t>(offset);
}

int decode_sign_magnitude15(std::uint16_t word)
{
  if (word & 0x8000U) {
    // Sign-magnitude payloads use bit 15 as the sign flag (0x7FFF magnitude),
    // matching the vendor SyncWriteSpe encoding and the proven decode paths.
    return -static_cast<int>(word & 0x7FFFU);
  }
  return static_cast<int>(word);
}

}  // namespace feetech
