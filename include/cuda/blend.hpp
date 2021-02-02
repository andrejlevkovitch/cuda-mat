// blend.hpp
/**\file
 */

#pragma once


#include "cuda/gpu_mat.hpp"


namespace cuda {
namespace detail {
__global__ void blend44(cuda::gpu_mat_ptr<uchar4> in_mat,
                        cuda::gpu_mat_ptr<uchar4> layer,
                        cuda::gpu_mat_ptr<uchar4> out_mat) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out_mat, row);
  GET_COL_OR_RETURN(out_mat, col);

  uchar4 l_pixel  = layer(row, col);
  uchar4 in_pixel = in_mat(row, col);

  if (l_pixel.w == 0) {
    out_mat(row, col) = in_pixel;
    return;
  }


  float beta  = l_pixel.w / 255.f;
  float alpha = (in_pixel.w / 255.f) * (1.f - beta);
  float out_a = beta + alpha;

  uchar4 out_pixel;
  out_pixel.x = (in_pixel.x * alpha + l_pixel.x * beta) / out_a;
  out_pixel.y = (in_pixel.y * alpha + l_pixel.y * beta) / out_a;
  out_pixel.z = (in_pixel.z * alpha + l_pixel.z * beta) / out_a;
  out_pixel.w = out_a * 255;

  out_mat(row, col) = out_pixel;
}

__global__ void blend34(cuda::gpu_mat_ptr<uchar3> in_mat,
                        cuda::gpu_mat_ptr<uchar4> layer,
                        cuda::gpu_mat_ptr<uchar3> out_mat) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out_mat, row);
  GET_COL_OR_RETURN(out_mat, col);

  uchar4 l_pixel  = layer(row, col);
  uchar3 in_pixel = in_mat(row, col);

  if (l_pixel.w == 0) {
    out_mat(row, col) = in_pixel;
    return;
  }


  float beta  = l_pixel.w / 255.f;
  float alpha = 1.f - beta;
  float out_a = beta + alpha;

  uchar3 out_pixel;
  out_pixel.x = (in_pixel.x * alpha + l_pixel.x * beta) / out_a;
  out_pixel.y = (in_pixel.y * alpha + l_pixel.y * beta) / out_a;
  out_pixel.z = (in_pixel.z * alpha + l_pixel.z * beta) / out_a;

  out_mat(row, col) = out_pixel;
}
} // namespace detail


/**\note it is safety to use same matrix as source and destination
 */
template <typename IOType, typename LayerType>
void blend(const cuda::gpu_mat<IOType> &   src,
           const cuda::gpu_mat<LayerType> &layer,
           cuda::gpu_mat<IOType> &         dst,
           const cuda::stream &            s = cuda::stream{0});

template <>
void blend<uchar4, uchar4>(const cuda::gpu_mat<uchar4> &src,
                           const cuda::gpu_mat<uchar4> &layer,
                           cuda::gpu_mat<uchar4> &      dst,
                           const cuda::stream &         s) {
  ASSERT_ARG(src.width() == layer.width() && src.height() == layer.height(),
             "sizes of src and layer not same");

  if (dst.width() != src.width() || dst.height() != src.height()) {
    dst = cuda::gpu_mat<uchar4>{src.height(), src.width(), s};
  }

  if (dst.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(dst);
  dim3 block = GET_BLOCK_DIM(dst);

  detail::blend44<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(src),
                                               cuda::make_gpu_mat_ptr(layer),
                                               cuda::make_gpu_mat_ptr(dst));
}

template <>
void blend<uchar3, uchar4>(const cuda::gpu_mat<uchar3> &src,
                           const cuda::gpu_mat<uchar4> &layer,
                           cuda::gpu_mat<uchar3> &      dst,
                           const cuda::stream &         s) {
  ASSERT_ARG(src.width() == layer.width() && src.height() == layer.height(),
             "sizes of src and layer not same");

  if (dst.width() != src.width() || dst.height() != src.height()) {
    dst = cuda::gpu_mat<uchar3>{src.height(), src.width(), s};
  }

  if (dst.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(dst);
  dim3 block = GET_BLOCK_DIM(dst);

  detail::blend34<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(src),
                                               cuda::make_gpu_mat_ptr(layer),
                                               cuda::make_gpu_mat_ptr(dst));
}
} // namespace cuda
