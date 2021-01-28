// info.cu

#include <cuda_runtime.h>
#include <iostream>

int main() {
  int            count;
  cudaDeviceProp prop;


  cudaGetDeviceCount(&count);
  std::cout << "devices count   : " << count << std::endl;


  cudaGetDeviceProperties(&prop, 0);
  std::cout << "max block size  : " << prop.maxGridSize[0] << ' '
            << prop.maxGridSize[1] << ' ' << prop.maxGridSize[2] << std::endl;
  std::cout << "max thread size : " << prop.maxThreadsDim[0] << ' '
            << prop.maxThreadsDim[1] << ' ' << prop.maxThreadsDim[2]
            << std::endl;

  return EXIT_SUCCESS;
}
