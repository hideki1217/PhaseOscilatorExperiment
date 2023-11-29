#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename... Args>
std::string format(const char *fmt, Args... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt, args...);
  return std::string(&buf[0], &buf[0] + len);
}

template <typename... Args>
void _info(const char *pos, int row_id, const char *fmt, Args... args) {
  const auto message = format(fmt, args...);
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::cout << std::put_time(localtime(&now_c), "[%Y/%m/%d %H:%M:%S] ")
            << "[INFO] " << message << " [" << pos << ":" << row_id << "]"
            << std::endl;
}
template <typename... Args>
void _worn(const char *pos, int row_id, const char *fmt, Args... args) {
  const auto message = format(fmt, args...);
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::cout << std::put_time(localtime(&now_c), "[%Y/%m/%d %H:%M:%S] ")
            << "[WORN] " << message << " [" << pos << ":" << row_id << "]"
            << std::endl;
}
#define log_info(...) _info(__FILE__, __LINE__, __VA_ARGS__)
#define log_worn(...) _worn(__FILE__, __LINE__, __VA_ARGS__)

#define eprintf(...) std::fprintf(stderr, __VA_ARGS__)

namespace lib {
namespace param {
template <typename Real>
void create_w(int ndim, Real *w) {
  const Real x0 = 0, r = 1.0;
  for (int i = 0; i < ndim; i++) {
    const Real p = Real(i + 1) / (ndim + 1);
    w[i] = x0 + r * std::tan(M_PI * (p - 0.5));
  }
}
}  // namespace param
}  // namespace lib
