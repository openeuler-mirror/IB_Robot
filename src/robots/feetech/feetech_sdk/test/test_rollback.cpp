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
#include <functional>
#include <set>
#include <utility>
#include <vector>

#include "feetech/detail/rollback.hpp"

namespace
{

using Calls = std::vector<std::pair<std::uint8_t, std::uint8_t>>;

TEST(Rollback, DisableTorqueOnAbortUsesReverseOrder)
{
  Calls calls;
  const bool disabled = feetech::detail::disable_torque_on_abort(
    {1, 2}, [&calls](std::uint8_t id, std::uint8_t enable) {
      calls.emplace_back(id, enable);
      return 1;
    });

  EXPECT_TRUE(disabled);
  ASSERT_EQ(calls.size(), 2U);
  EXPECT_EQ(calls[0], std::make_pair(std::uint8_t{2}, std::uint8_t{0}));
  EXPECT_EQ(calls[1], std::make_pair(std::uint8_t{1}, std::uint8_t{0}));
}

TEST(Rollback, DisableTorqueRetriesAndReportsFailure)
{
  int attempts = 0;
  const bool disabled = feetech::detail::disable_torque_on_abort(
    {1}, [&attempts](std::uint8_t, std::uint8_t) {
      ++attempts;
      return 0;
    });

  EXPECT_FALSE(disabled);
  EXPECT_EQ(attempts, 3);
}

TEST(Rollback, PartialConfigRelocksOnlyUnlockedMotors)
{
  // motor_ids {1, 2, 3}; only motor 2 was left unlocked at abort time.
  Calls torque_calls;
  std::vector<std::uint8_t> lock_calls;
  const auto result = feetech::detail::rollback_partial_config(
    {1, 2, 3}, std::set<std::uint8_t>{2},
    [&torque_calls](std::uint8_t id, std::uint8_t enable) {
      torque_calls.emplace_back(id, enable);
      return 1;
    },
    [&lock_calls](std::uint8_t id) {
      lock_calls.push_back(id);
      return 1;
    });

  EXPECT_TRUE(result.torque_disabled_all);
  EXPECT_TRUE(result.eprom_relocked_all);
  EXPECT_TRUE(result.relock_failures.empty());
  // Torque disable is fail-closed over every motor (reverse order).
  ASSERT_EQ(torque_calls.size(), 3U);
  EXPECT_EQ(torque_calls[0].first, std::uint8_t{3});
  EXPECT_EQ(torque_calls[2].first, std::uint8_t{1});
  // Only the unlocked motor is relocked.
  ASSERT_EQ(lock_calls.size(), 1U);
  EXPECT_EQ(lock_calls[0], std::uint8_t{2});
}

TEST(Rollback, RelockFailureIsReportedDistinctly)
{
  const auto result = feetech::detail::rollback_partial_config(
    {1}, std::set<std::uint8_t>{1},
    [](std::uint8_t, std::uint8_t) { return 1; },
    [](std::uint8_t) { return 0; });

  EXPECT_TRUE(result.torque_disabled_all);
  EXPECT_FALSE(result.eprom_relocked_all);
  ASSERT_EQ(result.relock_failures.size(), 1U);
  EXPECT_EQ(result.relock_failures[0], std::uint8_t{1});
}

}  // namespace
