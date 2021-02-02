// warp.hpp
/**\file
 */

#pragma once

#include "cuda/gpu_mat.hpp"

// XXX remove in future
#include <opencv2/core.hpp>


namespace cuda {
struct affine_matrix {
  float raw[6];
};

struct perspective_matrix {
  float raw[9];
};

namespace detail {
template <typename TransformationMatrix>
__device__ float2 transform(const TransformationMatrix &m, float2 point);

template <>
__device__ float2 transform<affine_matrix>(const affine_matrix &m,
                                           float2               point) {
  float2       output;
  const float *mat = m.raw;
  output.x         = mat[0] * point.x + mat[1] * point.y + mat[2];
  output.y         = mat[3] * point.x + mat[4] * point.y + mat[5];
  return output;
}

template <>
__device__ float2 transform<perspective_matrix>(const perspective_matrix &m,
                                                float2 point) {
  float2       output;
  const float *mat   = m.raw;
  float        coeff = 1.f / (mat[6] * point.x + mat[7] * point.y + mat[8]);
  output.x           = coeff * (mat[0] * point.x + mat[1] * point.y + mat[2]);
  output.y           = coeff * (mat[3] * point.x + mat[4] * point.y + mat[5]);
  return output;
}


template <typename IOType, typename TransformationMatrix>
__global__ void warp(const gpu_mat_ptr<IOType>  input,
                     gpu_mat_ptr<IOType>        output,
                     const TransformationMatrix transform_mat,
                     IOType                     border_val) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(output, row);
  GET_COL_OR_RETURN(output, col);

  float2 src_point =
      transform(transform_mat,
                make_float2(__uint2float_rd(col), __uint2float_rd(row)));

  // XXX we can't convert to uint values that less then 0
  if (src_point.x < 0 || src_point.y < 0) {
    output(row, col) = border_val;
    return;
  }

  unsigned int src_row = __float2uint_rd(src_point.y);
  unsigned int src_col = __float2uint_rd(src_point.x);
  if (src_row < input.height() && src_col < input.width()) {
    output(row, col) = input(src_row, src_col);
  } else {
    output(row, col) = border_val;
  }
}
} // namespace detail


template <typename IOType, typename TransformationMatrix>
void warp(const gpu_mat<IOType> &input,
          gpu_mat<IOType> &      output,
          TransformationMatrix   transform_mat,
          int                    flags        = 0,
          int                    border_mode  = 0,
          IOType                 border_value = 0,
          const stream &         s            = stream{0}) {
  ASSERT_ARG(sizeof(transform_mat.raw) / sizeof(float) <= 9,
             "invalid transformation matrix size");

  if (input.empty() || output.empty()) {
    return;
  }


  // at first inverse transformation matrix
  // clang-format off
  float tmp_mat[9] = {1, 0, 0,
                      0, 1, 0,
                      0, 0, 1};
  // clang-format on

  memcpy(tmp_mat, transform_mat.raw, sizeof(transform_mat.raw));

  // TODO remove usage of opencv method and change to own implementation
  cv::Mat t_cv_mat{3, 3, CV_32FC1, (void *)tmp_mat};
  t_cv_mat = t_cv_mat.inv();

  memcpy(transform_mat.raw, tmp_mat, sizeof(transform_mat.raw));


  // transformation
  detail::warp<<<GET_GRID_DIM(output), GET_BLOCK_DIM(output), 0, s.raw()>>>(
      make_gpu_mat_ptr(input),
      make_gpu_mat_ptr(output),
      transform_mat,
      border_value);
}


/**\brief affine transformation
 * \param output must be allocated with needed trenasformation size
 * \param transform_mat affine transformation matrix
 * \param flags reserved for future usage, now supports only nearest
 * interpolation
 * \param border_mode reserved for future usage, now supports only constant
 * border mode
 * \warning usage one matrix for input and output produce undefined behavour
 */
template <typename IOType>
void warpAffine(const gpu_mat<IOType> &input,
                gpu_mat<IOType> &      output,
                const affine_matrix &  transform_mat,
                int                    flags        = 0,
                int                    border_mode  = 0,
                IOType                 border_value = 0,
                const stream &         s            = stream{0}) {
  warp(input, output, transform_mat, flags, border_mode, border_value, s);
}


/**\brief perspective transformation
 * \param output must be allocated with needed trenasformation size
 * \param transform_mat perspective transformation matrix
 * \param flags reserved for future usage, now supports only nearest
 * interpolation
 * \param border_mode reserved for future usage, now supports only constant
 * border mode
 * \warning usage one matrix for input and output produce undefined behavour
 */
template <typename IOType>
void warpPerspective(const gpu_mat<IOType> &   input,
                     gpu_mat<IOType> &         output,
                     const perspective_matrix &transform_mat,
                     int                       flags        = 0,
                     int                       border_mode  = 0,
                     IOType                    border_value = 0,
                     const stream &            s            = stream{0}) {
  warp(input, output, transform_mat, flags, border_mode, border_value, s);
}
} // namespace cuda
