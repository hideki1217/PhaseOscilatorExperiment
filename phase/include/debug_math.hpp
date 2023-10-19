#pragma once

#include <iostream>

template<typename T>
void print_mtx(int n, const T* x) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << x[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

template<typename T>
void print_vec(int n, const T* x) {
  for (int i = 0; i < n; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
}