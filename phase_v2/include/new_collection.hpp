#pragma once

#include <valarray>
#include <vector>

namespace new_lib {
namespace collection {

template <typename T>
class FixedQueue {
 public:
  FixedQueue() {}
  FixedQueue(std::size_t size, T initial) : _sum(initial * size) {
    static_assert(sizeof(T) <= 8);

    _inner.resize(size, initial);
  }

  void resize(std::size_t size, T initial = 0) noexcept {
    _sum = size * initial;
    _inner.resize(size, initial);
  }

  void push(const T& x) noexcept {
    _sum -= _inner[_last];
    _inner[_last] = x;
    _sum += x;

    _last = (_last + 1) % size();
  }

  T first() noexcept { return _inner[_last]; }

  T sum() const noexcept { return _sum; }
  T mean() const noexcept { return _sum / size(); }
  std::size_t size() const noexcept { return _inner.size(); }

 private:
  std::vector<T> _inner;
  T _sum;
  int _last = 0;
};

template <typename T, int ndim>
class FixedQueueVec {
 public:
  FixedQueueVec() {}
  FixedQueueVec(std::size_t size) : _sum(ndim) {
    static_assert(sizeof(T) <= 8);

    resize(size);
  }

  void resize(std::size_t size) noexcept {
    _sum = static_cast<T>(0);
    _inner.resize(size, std::valarray<T>(static_cast<T>(0), ndim));
  }

  void push(const T* x) noexcept {
    _sum -= _inner[_last];
    std::copy_n(x, ndim, std::begin(_inner[_last]));
    _sum += _inner[_last];

    _last = (_last + 1) % size();
  }

  const std::valarray<T>& first() const noexcept { return _inner[_last]; }

  const std::valarray<T>& sum() const noexcept { return _sum; }
  std::size_t size() const noexcept { return _inner.size(); }

 private:
  std::vector<std::valarray<T>> _inner;
  std::valarray<T> _sum;
  int _last = 0;
};

}  // namespace collection
}  // namespace new_lib