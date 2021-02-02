// is_dublicates.cu

#include "cuda/arithmetic.hpp"
#include "cuda/gpu_mat.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>


#define THRESHOLD 5


namespace cuda {
namespace detail {
template <typename OType, typename IType>
__global__ void
pow(cuda::gpu_mat_ptr<IType> in, cuda::gpu_mat_ptr<OType> out, float power) {
  unsigned row = 0;
  unsigned col = 0;
  GET_ROW_OR_RETURN(in, row);
  GET_COL_OR_RETURN(in, col);

  out(row, col) = ::pow(in(row, col), power);
}
} // namespace detail

template <typename OType = float, typename IType>
cuda::gpu_mat<OType> pow(const cuda::gpu_mat<IType> &input,
                         float                       power,
                         const cuda::stream &        s = cuda::stream{0}) {
  cuda::gpu_mat<OType> output{input.height(), input.width(), s};

  if (input.empty()) {
    return output;
  }


  dim3 grid  = GET_GRID_DIM(input);
  dim3 block = GET_BLOCK_DIM(input);

  detail::pow<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(input),
                                           cuda::make_gpu_mat_ptr(output),
                                           power);

  return output;
}
} // namespace cuda


int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "you must set two arguments as names of image files"
              << std::endl;
    return EXIT_FAILURE;
  }


  std::string first_image_filename  = argv[1];
  std::string second_image_filename = argv[2];


  cv::Mat first_image = cv::imread(first_image_filename, cv::IMREAD_GRAYSCALE);
  cv::Mat second_image =
      cv::imread(second_image_filename, cv::IMREAD_GRAYSCALE);


  assert(first_image.type() == CV_8UC1);
  assert(second_image.type() == CV_8UC1);

  if (first_image.size() != second_image.size()) {
    std::cout << "diffirent sizes" << std::endl;
    return EXIT_FAILURE;
  }


  size_t height = first_image.rows;
  size_t width  = first_image.cols;
  size_t total  = height * width;


  cuda::gpu_mat<uchar> first_mat{height, width, first_image.data};
  cuda::gpu_mat<uchar> second_mat{height, width, second_image.data};


  cuda::gpu_mat<float> subtraction = cuda::subtract(first_mat, second_mat);
  cuda::gpu_mat<float> pow_mat     = cuda::pow(subtraction, 2);


  std::vector<float> pow_mat_data(total);
  pow_mat.download(pow_mat_data.data());


  double summ = 0;
  for (float val : pow_mat_data) {
    summ += val;
  }

  double diff = std::pow(summ / total, 0.5);


  if (diff > THRESHOLD) {
    std::cout << "don't looks as similar images, koaf =  " << diff << std::endl;

    return EXIT_FAILURE;
  }


  std::cout << "looks as similar images, koaf = " << diff << std::endl;

  return EXIT_SUCCESS;
}
