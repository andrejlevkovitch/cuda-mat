// arithmetic.hpp
/**\file
 * \warning be careful with usage signed integer values on gpu with calculation,
 * it can has undefined behaviour, because cuda has problem with signed integer
 * overflow
 */

#pragma once

#include "cuda/gpu_mat.hpp"
#include "cuda/misc.hpp"


namespace cuda {
namespace detail {
template <typename OType, typename IType>
__global__ void subtract(cuda::gpu_mat_ptr<IType> lhs,
                         cuda::gpu_mat_ptr<IType> rhs,
                         cuda::gpu_mat_ptr<OType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  OType first   = lhs(row, col);
  OType second  = rhs(row, col);
  out(row, col) = first - second;
}

template <typename IOType>
__global__ void add(cuda::gpu_mat_ptr<IOType> lhs,
                    cuda::gpu_mat_ptr<IOType> rhs,
                    cuda::gpu_mat_ptr<IOType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  out(row, col) = lhs(row, col) + rhs(row, col);
}

template <typename IOType>
__global__ void divide(cuda::gpu_mat_ptr<IOType> lhs,
                       cuda::gpu_mat_ptr<IOType> rhs,
                       cuda::gpu_mat_ptr<IOType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  IOType divider = rhs(row, col);
  if (divider != 0) {
    out(row, col) = lhs(row, col) / rhs(row, col);
  } else {
    out(row, col) = 0;
  }
}

template <typename IOType>
__global__ void multiply(cuda::gpu_mat_ptr<IOType> lhs,
                         cuda::gpu_mat_ptr<IOType> rhs,
                         cuda::gpu_mat_ptr<IOType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  out(row, col) = lhs(row, col) * rhs(row, col);
}
} // namespace detail


template <typename OType = float, typename IType>
cuda::gpu_mat<OType> subtract(const cuda::gpu_mat<IType> &lhs,
                              const cuda::gpu_mat<IType> &rhs,
                              const cuda::stream &        s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  cuda::gpu_mat<OType> out{lhs.height(), lhs.width(), s};
  if (out.empty()) {
    return out;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::subtract<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                                cuda::make_gpu_mat_ptr(rhs),
                                                cuda::make_gpu_mat_ptr(out));

  return out;
}

template <typename IOType>
cuda::gpu_mat<IOType> add(const cuda::gpu_mat<IOType> &lhs,
                          const cuda::gpu_mat<IOType> &rhs,
                          const cuda::stream &         s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  cuda::gpu_mat<IOType> out{lhs.height(), lhs.width(), s};
  if (out.empty()) {
    return out;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::add<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                           cuda::make_gpu_mat_ptr(rhs),
                                           cuda::make_gpu_mat_ptr(out));

  return out;
}

template <typename IOType>
cuda::gpu_mat<IOType> multiply(const cuda::gpu_mat<IOType> &lhs,
                               const cuda::gpu_mat<IOType> &rhs,
                               const cuda::stream &s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  cuda::gpu_mat<IOType> out{lhs.height(), lhs.width(), s};
  if (out.empty()) {
    return out;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::multiply<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                                cuda::make_gpu_mat_ptr(rhs),
                                                cuda::make_gpu_mat_ptr(out));

  return out;
}

/**\note safe to divide on zero
 */
template <typename IOType>
cuda::gpu_mat<IOType> divide(const cuda::gpu_mat<IOType> &lhs,
                             const cuda::gpu_mat<IOType> &rhs,
                             const cuda::stream &         s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  cuda::gpu_mat<IOType> out{lhs.height(), lhs.width(), s};
  if (out.empty()) {
    return out;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::divide<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                              cuda::make_gpu_mat_ptr(rhs),
                                              cuda::make_gpu_mat_ptr(out));

  return out;
}
} // namespace cuda
