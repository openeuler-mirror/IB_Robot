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

#include "feetech/detail/rollback.hpp"

namespace feetech
{
namespace detail
{

bool disable_torque_on_abort(
  const std::vector<std::uint8_t> & motor_ids,
  const std::function<int(std::uint8_t, std::uint8_t)> & enable_torque,
  int retry_count)
{
  bool all_disabled = true;
  for (auto it = motor_ids.rbegin(); it != motor_ids.rend(); ++it) {
    bool disabled = false;
    for (int attempt = 0; attempt < retry_count && !disabled; ++attempt) {
      disabled = enable_torque(*it, 0) != 0;
    }
    if (!disabled) {
      all_disabled = false;
    }
  }
  return all_disabled;
}

RollbackResult rollback_partial_config(
  const std::vector<std::uint8_t> & motor_ids,
  const std::set<std::uint8_t> & unlocked_motors,
  const std::function<int(std::uint8_t, std::uint8_t)> & enable_torque,
  const std::function<int(std::uint8_t)> & lock_eprom, int retry_count)
{
  RollbackResult result;
  result.torque_disabled_all = disable_torque_on_abort(motor_ids, enable_torque, retry_count);
  for (auto it = unlocked_motors.rbegin(); it != unlocked_motors.rend(); ++it) {
    if (lock_eprom(*it) == 0) {
      result.eprom_relocked_all = false;
      result.relock_failures.push_back(*it);
    }
  }
  return result;
}

}  // namespace detail
}  // namespace feetech
