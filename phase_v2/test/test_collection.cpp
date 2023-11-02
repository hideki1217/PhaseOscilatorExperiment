#include <cassert>
#include <collection.hpp>
#include <vector>

void test_FixedQueue() {
  auto q = lib::collection::FixedQueue<int>(3, 0);
  assert(q.sum() == 0);
  assert(q.size() == 3);

  assert(q.push(1) == 0);
  assert(q.sum() == 1);
  assert(q.push(2) == 0);
  assert(q.sum() == 3);
  assert(q.push(3) == 0);
  assert(q.sum() == 6);
  assert(q.push(4) == 1);
  assert(q.sum() == 9);

  q.resize(4);
  assert(q.sum() == 0);
}

int main() { test_FixedQueue(); }