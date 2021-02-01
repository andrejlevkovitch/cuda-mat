// warp_affine.cu

#include "warp.hpp"
#include <iostream>
#include <vector>

int main() {
  std::vector<float>         data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  const cuda::gpu_mat<float> input{3, 3, data.data()};
  cuda::gpu_mat<float>       output{4, 4, 0.};

  cuda::affine_matrix transform{{1, 0, 1, 0, 1, 1}};


  warpAffine(input, output, transform, 0, 0, 9.f);


  std::vector<float> out(output.total());
  output.download(out.data());

  for (size_t row = 0; row < output.height(); ++row) {
    for (size_t col = 0; col < output.width(); ++col) {
      size_t offset = row * output.width() + col;
      std::cout << out[offset] << ' ';
    }
    std::cout << std::endl;
  }


  return EXIT_SUCCESS;
}
