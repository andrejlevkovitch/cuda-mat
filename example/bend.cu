// bend.cu

#include "cuda/bend.hpp"
#include "cuda/remap.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>


#define TO_RAD(deg) ((deg) * (M_PI / 180.))


int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr
        << "set image filename as first argument and output filename as second"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::string input_filename  = argv[1];
  std::string output_filename = argv[2];


  cv::Mat input_image = cv::imread(input_filename, cv::IMREAD_COLOR);
  assert(input_image.type() == CV_8UC3);


  // get bending map
  uint2  size   = make_uint2(input_image.cols, input_image.rows);
  float2 anchor = make_float2(size.x / 2., size.y / 2.);
  float  angle  = TO_RAD(75);

  std::cout << size.x << ' ' << size.y << std::endl;

  cuda::gpu_mat<float2> bend_map = cuda::bend_map(size, anchor, angle);

  std::cout << bend_map.width() << ' ' << bend_map.height() << std::endl;


  // remap input matrix by bending map
  cuda::gpu_mat<uchar3> input_mat{size.y, size.x, input_image.data};
  cuda::gpu_mat<uchar3> output_mat{size.y, size.x};

  cuda::remap(input_mat, output_mat, bend_map);


  // save image
  cv::Mat output_image{input_image.size(), CV_8UC3};
  output_mat.download(output_image.data);

  cv::imwrite(output_filename, output_image);


  return EXIT_SUCCESS;
}
