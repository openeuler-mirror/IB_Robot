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

#ifndef FEETECH__SAFE_SMS_STS_HPP_
#define FEETECH__SAFE_SMS_STS_HPP_

#include <chrono>

#include "SMS_STS.h"

namespace feetech
{

/// SMS_STS with hardened serial reads, ported from the proven SO-101
/// SafeSMSSTS: rejects reads on an invalid descriptor without touching the
/// caller's buffer, handles closed peers, and honors the caller's timeout.
class SafeSmsSts : public SMS_STS
{
public:
  ~SafeSmsSts() { end(); }
  void end();
  bool write_ok() const { return write_ok_; }
  /// Bound all subsequent reads and writes by a shared cycle deadline until cleared.
  void set_io_deadline(std::chrono::steady_clock::time_point deadline)
  {
    io_deadline_ = deadline;
  }
  void clear_io_deadline()
  {
    io_deadline_ = std::chrono::steady_clock::time_point::max();
  }

protected:
  int readSCS(unsigned char * data, int length) override;
  int readSCS(unsigned char * data, int length, unsigned long timeout_ms) override;
  void wFlushSCS() override;

private:
  int read_with_timeout(unsigned char * data, int length, unsigned long timeout_ms);
  bool write_ok_ = true;
  std::chrono::steady_clock::time_point io_deadline_ =
    std::chrono::steady_clock::time_point::max();
};

}  // namespace feetech

#endif  // FEETECH__SAFE_SMS_STS_HPP_
