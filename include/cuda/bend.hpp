// bend.hpp
/**\file
 */

#include "cuda/gpu_mat.hpp"
#include "cuda/misc.hpp"
#include <cmath>


namespace cuda {
namespace detail {
__device__ float2 operator-(float2 first, float2 second) {
  return make_float2(first.x - second.x, first.y - second.y);
}

__global__ void fill_vec_map(cuda::gpu_mat_ptr<float2> vec_map) {
  unsigned int row = 0;
  unsigned int col = 0;

  GET_ROW_OR_RETURN(vec_map, row);
  GET_COL_OR_RETURN(vec_map, col);

  vec_map(row, col) = make_float2(col, row);
}

__global__ void bend(cuda::gpu_mat_ptr<float2> input_vec_map,
                     cuda::gpu_mat_ptr<float2> output_vec_map,
                     float2                    anchor,
                     float                     angle,
                     float2                    xy_factors) {
  unsigned int row = 0;
  unsigned int col = 0;

  GET_ROW_OR_RETURN(output_vec_map, row);
  GET_COL_OR_RETURN(output_vec_map, col);


  float2 to_anchor     = anchor - input_vec_map(row, col);
  float  atan          = atan2(to_anchor.y, to_anchor.x);
  float  to_anchor_len = norm3d(to_anchor.x, to_anchor.y, 0);

  float2 bend_norm;
  if (abs(atan) < angle * 2) { // swirl transform
    bend_norm = make_float2(cos(atan / 2.f), sin(atan / 2.f));
  } else { // simple affine rotation
    float sign = (atan) > 0 ? -1 : +1;

    bend_norm = make_float2(cos(atan + sign * angle), sin(atan + sign * angle));
  }

  float2 out_vec;
  out_vec.x = (1.f / xy_factors.x) * to_anchor_len * bend_norm.x;
  out_vec.y = (1.f / xy_factors.y) * to_anchor_len * bend_norm.y;

  output_vec_map(row, col) = anchor - out_vec;
}
} // namespace detail

/**\return map matrix for bending matricies around anchor in x-axis direction
 * \param angle angle for beding in radians
 * \param xy_factors x and y scale factors
 * \note angle for bending can't be more then 90 deg
 */
cuda::gpu_mat<float2> bend_map(uint2  mat_size,
                               float2 anchor,
                               float  angle,
                               float2 xy_factors     = make_float2(1.f, 1.f),
                               const cuda::stream &s = cuda::stream{0}) {
  ASSERT_ARG(xy_factors.x != 0 && xy_factors.y != 0,
             "xy_factors can't be a zero");
  ASSERT_ARG(angle < (M_PI / 2.) + 0.001,
             "bending angle can't be more then 90 deg");

  if (mat_size.x == 0 || mat_size.y == 0) {
    return cuda::gpu_mat<float2>{0, 0, s}; // do nothing
  }


  // at first fill input vector map
  cuda::gpu_mat<float2> input_vec_map{mat_size.y, mat_size.x, s};

  dim3 grid   = GET_GRID_DIM(input_vec_map);
  dim3 blocks = GET_BLOCK_DIM(input_vec_map);

  detail::fill_vec_map<<<grid, blocks, 0, s.raw()>>>(
      cuda::make_gpu_mat_ptr(input_vec_map));


  // calculate result bend map
  cuda::gpu_mat<float2> bend_map{mat_size.y, mat_size.x, s};

  // invert angle for calculation
  float bend_angle = M_PI / 2. - angle;

  detail::bend<<<grid, blocks, 0, s.raw()>>>(
      cuda::make_gpu_mat_ptr(input_vec_map),
      cuda::make_gpu_mat_ptr(bend_map),
      anchor,
      bend_angle,
      xy_factors);

  return bend_map;
}
} // namespace cuda
