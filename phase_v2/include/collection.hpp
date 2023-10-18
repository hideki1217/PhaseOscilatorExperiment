#include <vector>

namespace lib {
namespace collection {

template <typename T>
class FixedQueue {
 public:
  FixedQueue(size_t size, T initial) : _sum(initial * size) {
    _inner.resize(size, initial);
  }

  T push(T x) {
    auto last_value = _inner[last];
    _inner[last] = x;

    last = (last + 1) % size();
    _sum += x - last_value;
    return last_value;
  }

  T sum() { return _sum; }
  T size() { return _inner.size(); }

 private:
  std::vector<T> _inner;
  T _sum;
  int last = 0;
};

}  // namespace collection
}  // namespace lib