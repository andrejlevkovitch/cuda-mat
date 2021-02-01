// remap.cu

#include "cuda/remap.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>


int main() {
  std::vector<float>         data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  const cuda::gpu_mat<float> input{3, 3, data.data()};
  cuda::gpu_mat<float>       output{3, 3, 0.};

  std::vector<int>         x_data{2, 1, 0, 2, 1, 0, 2, 1, 0};
  std::vector<int>         y_data{2, 2, 2, 1, 1, 1, 0, 0, 0};
  const cuda::gpu_mat<int> map_x{3, 3, x_data.data()};
  const cuda::gpu_mat<int> map_y{3, 3, y_data.data()};


  remap(input, output, map_x, map_y);


  std::vector<float> out(output.total());
  output.download(out.data());

  for (float val : out) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;


  return EXIT_SUCCESS;
}
