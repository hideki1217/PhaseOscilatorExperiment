#include <cassert>
#include <new_collection.hpp>
#include <vector>

void test_FixedQueue() {
  auto q = new_lib::collection::FixedQueue<int>(3, 0);
  assert(q.sum() == 0);
  assert(q.size() == 3);

  assert(q.sum() == 1);
  q.push(2);
  assert(q.sum() == 3);
  q.push(3);
  assert(q.sum() == 6);
  q.push(4);
  assert(q.sum() == 9);

  q.resize(4);
  assert(q.sum() == 0);
}

int main() { test_FixedQueue(); }