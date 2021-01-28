// remap_async.cu

#include "remap.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>


int main() {
  cuda::stream s;

  std::vector<float>         data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  const cuda::gpu_mat<float> input{3, 3, data.data(), s};
  cuda::gpu_mat<float>       output{3, 3, 0., s};

  std::vector<int>         x_data{2, 1, 0, 2, 1, 0, 2, 1, 0};
  std::vector<int>         y_data{2, 2, 2, 1, 1, 1, 0, 0, 0};
  const cuda::gpu_mat<int> map_x{3, 3, x_data.data(), s};
  const cuda::gpu_mat<int> map_y{3, 3, y_data.data(), s};


  remap(input, output, map_x, map_y, 0, 0, 0.f, s);


  output.download(s);

  s.synchronize();


  std::vector<float> out(output.total());
  memcpy(out.data(), output.host_ptr(), output.bytes());

  for (float val : out) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;


  return EXIT_SUCCESS;
}
