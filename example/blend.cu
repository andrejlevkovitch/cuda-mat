// blend.cu

#include "cuda/blend.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


int main(int argc, const char *argv[]) {
  if (argc != 4) {
    std::cerr << "you must set three arguments: src, layer and dst filenames"
              << std::endl;
    return EXIT_FAILURE;
  }


  std::string src_image_filename   = argv[1];
  std::string layer_image_filename = argv[2];
  std::string dst_image_filename   = argv[3];


  cv::Mat src_image   = cv::imread(src_image_filename, cv::IMREAD_UNCHANGED);
  cv::Mat layer_image = cv::imread(layer_image_filename, cv::IMREAD_UNCHANGED);


  assert(src_image.type() == CV_8UC3 || src_image.type() == CV_8UC4);
  assert(layer_image.type() == CV_8UC4);


  if (src_image.size() != layer_image.size()) {
    std::cout << "diffirent sizes" << std::endl;
    return EXIT_FAILURE;
  }


  size_t height = src_image.rows;
  size_t width  = src_image.cols;


  cv::Mat dst_image{src_image.size(), src_image.type()};
  switch (src_image.type()) {
  case CV_8UC3: {
    cuda::gpu_mat<uchar3> src_mat{height, width, src_image.data};
    cuda::gpu_mat<uchar4> layer_mat{height, width, layer_image.data};
    cuda::gpu_mat<uchar3> dst_mat{height, width};

    cuda::blend(src_mat, layer_mat, dst_mat);

    dst_mat.download(dst_image.data);
  } break;
  case CV_8UC4:
    cuda::gpu_mat<uchar4> src_mat{height, width, src_image.data};
    cuda::gpu_mat<uchar4> layer_mat{height, width, layer_image.data};
    cuda::gpu_mat<uchar4> dst_mat{height, width};

    cuda::blend(src_mat, layer_mat, dst_mat);

    dst_mat.download(dst_image.data);
    break;
  }


  cv::imwrite(dst_image_filename, dst_image);

  return EXIT_SUCCESS;
}
