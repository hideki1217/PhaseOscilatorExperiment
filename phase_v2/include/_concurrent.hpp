#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace lib {
namespace concurrent {

class ThreadPool {
 public:
  ThreadPool(std::size_t size) : terminated(false) {
    for (std::size_t i = 0; i < size; ++i) {
      workers.emplace_back([this] { spawn(); });
    }
  }

  virtual ~ThreadPool() { terminate(); }

  void terminate() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      terminated = true;
    }
    condition.notify_all();

    for (std::size_t i = 0; i < workers.size(); ++i) {
      workers[i].join();
    }
  }

  void post(std::function<void()> f) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      pendings.push(f);
    }

    condition.notify_one();
  }

  void join() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      owner_condition.wait(
          lock, [this] { return pendings.empty() && runnings == 0; });
    }
  }

 private:
  void spawn() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock,
                       [this] { return terminated || !pendings.empty(); });
        if (terminated && pendings.empty()) {
          return;
        }

        task = std::move(pendings.front());
        pendings.pop();
        ++runnings;
      }
      task();
      {
        std::unique_lock<std::mutex> lock(mutex);
        --runnings;
        owner_condition.notify_one();
      }
    }
  }

  std::vector<std::thread> workers;
  std::queue<std::function<void()>> pendings;
  int runnings;

  std::mutex mutex;
  std::condition_variable condition;
  std::condition_variable owner_condition;
  bool terminated;
};
}  // namespace concurrent
}  // namespace lib
