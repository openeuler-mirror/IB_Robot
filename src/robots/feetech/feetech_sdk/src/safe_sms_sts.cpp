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

#include "safe_sms_sts.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <poll.h>
#include <time.h>
#include <unistd.h>

namespace feetech
{

void SafeSmsSts::end()
{
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
  txBufLen = 0;
}

void SafeSmsSts::wFlushSCS()
{
  write_ok_ = fd >= 0;
  int sent = 0;
  const auto deadline = std::min(
    std::chrono::steady_clock::now() + std::chrono::milliseconds(IOTimeOut), io_deadline_);
  while (write_ok_ && sent < txBufLen) {
    if (std::chrono::steady_clock::now() >= deadline) {
      write_ok_ = false;
      break;
    }
    const auto count = ::write(fd, txBuf + sent, txBufLen - sent);
    if (count > 0) {
      sent += static_cast<int>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      // Preserve the unsent suffix while waiting; EINTR never restarts the budget.
      while (write_ok_) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::steady_clock::duration::zero()) {
          write_ok_ = false;
          break;
        }
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(remaining);
        const timespec timeout{
          static_cast<time_t>(seconds.count()),
          static_cast<long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            remaining - seconds).count())};
        pollfd descriptor{fd, POLLOUT, 0};
        const int ready = ::ppoll(&descriptor, 1, &timeout, nullptr);
        if (ready < 0 && errno == EINTR) {
          continue;
        }
        write_ok_ = ready > 0 && (descriptor.revents & POLLOUT) != 0 &&
          (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) == 0;
        break;
      }
    } else {
      write_ok_ = false;
    }
  }
  txBufLen = 0;
}

int SafeSmsSts::readSCS(unsigned char * data, int length)
{
  return read_with_timeout(data, length, IOTimeOut);
}

int SafeSmsSts::readSCS(unsigned char * data, int length, unsigned long timeout_ms)
{
  return read_with_timeout(data, length, timeout_ms);
}

int SafeSmsSts::read_with_timeout(
  unsigned char * data, int length, unsigned long timeout_ms)
{
  if (data == nullptr || length < 0 || fd < 0) {
    return -1;
  }
  if (length == 0) {
    return 0;
  }

  const auto deadline = std::min(
    std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms), io_deadline_);
  int received = 0;
  while (received < length) {
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
      return received;
    }
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(remaining);
    const timespec timeout{
      static_cast<time_t>(seconds.count()),
      static_cast<long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        remaining - seconds).count())};
    pollfd descriptor{fd, POLLIN, 0};
    const int selected = ::ppoll(&descriptor, 1, &timeout, nullptr);
    if (selected == 0) {
      return received;
    }
    if (selected < 0) {
      if (errno == EINTR) {
        continue;
      }
      return received > 0 ? received : -1;
    }
    if ((descriptor.revents & POLLNVAL) != 0) {
      return received > 0 ? received : -1;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return received;
    }

    const ssize_t count =
      ::read(fd, data + received, static_cast<size_t>(length - received));
    if (count > 0) {
      received += static_cast<int>(count);
      continue;
    }
    if (count == 0) {
      return received;
    }
    if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      return received > 0 ? received : -1;
    }
  }
  return received;
}

}  // namespace feetech
