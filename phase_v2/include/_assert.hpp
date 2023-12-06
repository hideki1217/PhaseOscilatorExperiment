#pragma once

#include <chrono>
#include <cmath>
#include <iostream>

static void _assert_eq(double actual, double respect, double acc,
                       const char* str_actual, const char* str_respect,
                       const char* file, const int line) {
  if (std::abs(actual - respect) > acc) {
    std::cerr << file << ":" << line << ": |" << str_actual << " - "
              << str_respect << "| < " << acc << std::endl;
    std::cerr << "    Actual  : " << actual << std::endl;
    std::cerr << "    Respect : " << respect << std::endl;
    std::exit(1);
  }
}
#define assert_nearly_eq(actual, respect, acc) \
  _assert_eq(actual, respect, acc, #actual, #respect, __FILE__, __LINE__)