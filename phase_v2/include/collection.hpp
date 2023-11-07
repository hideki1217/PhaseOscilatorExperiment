#pragma once

#include <vector>

namespace lib {
namespace collection {

template <typename T>
class FixedQueue {
 public:
  FixedQueue() {}
  FixedQueue(std::size_t size, T initial) : _sum(initial * size) {
    _inner.resize(size, initial);
  }

  void resize(std::size_t size, T initial = 0) noexcept {
    _sum = size * initial;
    _inner.resize(size, initial);
  }

  T push(T x) noexcept {
    auto last_value = _inner[last];
    _inner[last] = x;

    last = (last + 1) % size();
    _sum += x - last_value;
    return last_value;
  }

  T sum() const noexcept { return _sum; }
  T mean() const noexcept { return _sum / size(); }
  std::size_t size() const noexcept { return _inner.size(); }

 private:
  std::vector<T> _inner;
  T _sum;
  int last = 0;
};

}  // namespace collection
}  // namespace lib