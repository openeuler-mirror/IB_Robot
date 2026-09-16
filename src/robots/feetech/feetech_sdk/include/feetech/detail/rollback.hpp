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

#ifndef FEETECH__DETAIL__ROLLBACK_HPP_
#define FEETECH__DETAIL__ROLLBACK_HPP_

#include <cstdint>
#include <functional>
#include <set>
#include <vector>

namespace feetech
{
namespace detail
{

/// Disable torque on every motor, retrying each motor `retry_count` times.
/// Returns true only when every motor acknowledged the disable.
bool disable_torque_on_abort(
  const std::vector<std::uint8_t> & motor_ids,
  const std::function<int(std::uint8_t, std::uint8_t)> & enable_torque,
  int retry_count = 3);

/// Outcome of a configuration rollback. The two success flags are independent
/// so callers can surface EPROM relock failures distinctly from torque
/// disable failures while keeping torque-off fail-closed semantics.
struct RollbackResult
{
  /// True only when every motor's torque was disabled.
  bool torque_disabled_all{true};
  /// True only when every motor left unlocked was successfully relocked.
  bool eprom_relocked_all{true};
  /// Motor IDs whose EPROM could not be relocked (unlocked at abort time).
  std::vector<std::uint8_t> relock_failures;
};

/// Roll back a partial configuration: fail-closed torque disable for every
/// motor (reverse order), followed by a best-effort EPROM relock for motors
/// tracked as unlocked.
[[nodiscard]] RollbackResult rollback_partial_config(
  const std::vector<std::uint8_t> & motor_ids,
  const std::set<std::uint8_t> & unlocked_motors,
  const std::function<int(std::uint8_t, std::uint8_t)> & enable_torque,
  const std::function<int(std::uint8_t)> & lock_eprom, int retry_count = 3);

}  // namespace detail
}  // namespace feetech

#endif  // FEETECH__DETAIL__ROLLBACK_HPP_
