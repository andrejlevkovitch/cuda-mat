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

template <typename OType, typename IType>
__global__ void add(cuda::gpu_mat_ptr<IType> lhs,
                    cuda::gpu_mat_ptr<IType> rhs,
                    cuda::gpu_mat_ptr<OType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  OType first   = lhs(row, col);
  OType second  = rhs(row, col);
  out(row, col) = first + second;
}

template <typename OType, typename IType>
__global__ void divide(cuda::gpu_mat_ptr<IType> lhs,
                       cuda::gpu_mat_ptr<IType> rhs,
                       cuda::gpu_mat_ptr<OType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  OType divider = rhs(row, col);
  if (divider != 0) {
    OType first   = lhs(row, col);
    out(row, col) = first / divider;
  } else {
    out(row, col) = 0;
  }
}

template <typename OType, typename IType>
__global__ void multiply(cuda::gpu_mat_ptr<IType> lhs,
                         cuda::gpu_mat_ptr<IType> rhs,
                         cuda::gpu_mat_ptr<OType> out) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(out, row);
  GET_COL_OR_RETURN(out, col);

  OType first   = lhs(row, col);
  OType second  = rhs(row, col);
  out(row, col) = first * second;
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

template <typename OType = float, typename IType>
cuda::gpu_mat<OType> add(const cuda::gpu_mat<IType> &lhs,
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

  detail::add<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                           cuda::make_gpu_mat_ptr(rhs),
                                           cuda::make_gpu_mat_ptr(out));

  return out;
}

template <typename OType = float, typename IType>
cuda::gpu_mat<OType> multiply(const cuda::gpu_mat<IType> &lhs,
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

  detail::multiply<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                                cuda::make_gpu_mat_ptr(rhs),
                                                cuda::make_gpu_mat_ptr(out));

  return out;
}

/**\note safe to divide on zero
 */
template <typename OType = float, typename IType>
cuda::gpu_mat<OType> divide(const cuda::gpu_mat<IType> &lhs,
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

  detail::divide<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                              cuda::make_gpu_mat_ptr(rhs),
                                              cuda::make_gpu_mat_ptr(out));

  return out;
}


/**\note it is safe to use one matrix for input and output
 */
template <typename OType, typename IType>
void subtract(const cuda::gpu_mat<IType> &lhs,
              const cuda::gpu_mat<IType> &rhs,
              cuda::gpu_mat<OType> &      out,
              const cuda::stream &        s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");

  if (lhs.height() != out.height() || lhs.width() != out.width()) {
    out = cuda::gpu_mat<OType>{lhs.height(), lhs.width(), s};
  }

  if (out.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::subtract<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                                cuda::make_gpu_mat_ptr(rhs),
                                                cuda::make_gpu_mat_ptr(out));
}

/**\note it is safe to use one matrix for input and output
 */
template <typename OType, typename IType>
void add(const cuda::gpu_mat<IType> &lhs,
         const cuda::gpu_mat<IType> &rhs,
         cuda::gpu_mat<OType> &      out,
         const cuda::stream &        s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  if (lhs.height() != out.height() || lhs.width() != out.width()) {
    out = cuda::gpu_mat<OType>{lhs.height(), lhs.width(), s};
  }

  if (out.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::add<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                           cuda::make_gpu_mat_ptr(rhs),
                                           cuda::make_gpu_mat_ptr(out));
}

/**\note it is safe to use one matrix for input and output
 */
template <typename OType, typename IType>
void multiply(const cuda::gpu_mat<IType> &lhs,
              const cuda::gpu_mat<IType> &rhs,
              cuda::gpu_mat<OType> &      out,
              const cuda::stream &        s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  if (lhs.height() != out.height() || lhs.width() != out.width()) {
    out = cuda::gpu_mat<OType>{lhs.height(), lhs.width(), s};
  }

  if (out.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::multiply<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                                cuda::make_gpu_mat_ptr(rhs),
                                                cuda::make_gpu_mat_ptr(out));
}

/**\note safe to divide on zero
 * \note it is safe to use one matrix for input and output
 */
template <typename OType, typename IType>
void divide(const cuda::gpu_mat<IType> &lhs,
            const cuda::gpu_mat<IType> &rhs,
            cuda::gpu_mat<OType> &      out,
            const cuda::stream &        s = cuda::stream{0}) {
  ASSERT_ARG(lhs.height() == rhs.height() && lhs.width() == rhs.width(),
             "matrix sizes are not same");


  if (lhs.height() != out.height() || lhs.width() != out.width()) {
    out = cuda::gpu_mat<OType>{lhs.height(), lhs.width(), s};
  }

  if (out.empty()) {
    return;
  }


  dim3 grid  = GET_GRID_DIM(lhs);
  dim3 block = GET_BLOCK_DIM(lhs);

  detail::divide<<<grid, block, 0, s.raw()>>>(cuda::make_gpu_mat_ptr(lhs),
                                              cuda::make_gpu_mat_ptr(rhs),
                                              cuda::make_gpu_mat_ptr(out));
}
} // namespace cuda
