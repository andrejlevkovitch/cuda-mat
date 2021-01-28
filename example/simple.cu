// simple.cu

#include "gpu_mat.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>


template <typename T>
__global__ void add_one(cuda::gpu_mat_ptr<T> mat) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  mat(row, col) += 1;
}


int main() {
  cuda::gpu_mat<float> mat{1, 10, 5.};
  // XXX or
  // std::vector<float> vec(10, 5);
  // cuda::gpu_mat<float> mat{1, 10, vec.data()};

  add_one<<<mat.height(), mat.width()>>>(make_gpu_mat_ptr(mat));

  std::vector<float> out(10);
  mat.download(out.data());

  for (float val : out) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
