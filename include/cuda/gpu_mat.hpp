// gpu_mat.hpp
/**\file
 */

#pragma once

#include "cuda/exception.hpp"
#include "cuda/stream.hpp"
#include "misc.hpp"
#include <cstddef>
#include <cuda_runtime.h>


#define BLOCK_X           32
#define BLOCK_Y           32
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y)
#define GRID_Y            32

/**\brief calculate optimal grid for using in cuda kernel
 * \note capacity of blocks can be more then capacity of rows in matrix
 */
#define GET_GRID_DIM(mat)                                                      \
  dim3((mat).total() / (GRID_Y * THREADS_PER_BLOCK) +                          \
           ((mat).total() % (GRID_Y * THREADS_PER_BLOCK) ? 1 : 0),             \
       GRID_Y)
/**\brief calculate optimal block size for using in cuda kernel
 * \note capacity of blocks can be more then capacity of rows in matrix
 */
#define GET_BLOCK_DIM(mat) dim3(BLOCK_X, BLOCK_Y)

/**\return number of current block
 * \note for gpu usage only
 */
#define GET_BLOCK() (blockIdx.y * gridDim.x + blockIdx.x)
/**\return number of current thread
 * \note for gpu usage only
 */
#define GET_THREAD() (threadIdx.y * blockDim.x + threadIdx.x)


#define GET_ROW(mat_width)                                                     \
  ((GET_BLOCK() * THREADS_PER_BLOCK + GET_THREAD()) / mat_width)
#define GET_COL(mat_width)                                                     \
  ((GET_BLOCK() * THREADS_PER_BLOCK + GET_THREAD()) % mat_width)


/**\brief help macros, uses for check that row is inside the matrix, if row is
 * outside of the matrix, then return from function
 * \param mat gpu_mat_ptr
 * \param row_var variable that will contain current row
 * \note for gpu usage only
 */
#define GET_ROW_OR_RETURN(mat, row_var)                                        \
  row_var = GET_ROW(mat.width());                                              \
  if (row_var >= mat.height()) {                                               \
    return;                                                                    \
  }
/**\brief help macros, uses for check that col is inside the matrix, if col is
 * outside of the matrix, then return from function
 * \param mat gpu_mat_ptr
 * \param row_var variable that will contain current col
 * \note for gpu usage only
 */
#define GET_COL_OR_RETURN(mat, col_var)                                        \
  col_var = GET_COL(mat.width());                                              \
  if (col_var >= mat.width()) {                                                \
    return;                                                                    \
  }


namespace cuda {

namespace detail {
template <typename T>
__global__ void fill(T *ptr, unsigned int height, unsigned int width, T val) {
  unsigned int row = GET_ROW(width);
  unsigned int col = GET_COL(width);

  if (row >= height || col >= width) {
    return;
  }

  unsigned int offset = row * width + col;
  ptr[offset]         = val;
}
} // namespace detail

template <typename T>
/**\note you can't use it directly in cuda code, use gpu_mat_ptr instead
 */
class gpu_mat {
public:
  using value_type = T;

  gpu_mat(size_t height, size_t width, void *data, const stream &s = stream{0})
      : host_{nullptr}
      , dev_{nullptr}
      , height_{height}
      , width_{width} {
    if (this->empty()) {
      return;
    }

    try {
      size_t bytes = this->bytes();
      THROW_IF_ERR(cudaMalloc(&dev_, bytes));

      if (s.is_default() == false) {
        cudaHostAlloc(&host_, bytes, cudaHostAllocDefault);
      }

      if (data != nullptr) {
        if (s.is_default()) {
          THROW_IF_ERR(cudaMemcpy(dev_, data, bytes, cudaMemcpyHostToDevice));
        } else {
          memcpy(host_, data, bytes);

          THROW_IF_ERR(cudaMemcpyAsync(dev_,
                                       host_,
                                       bytes,
                                       cudaMemcpyHostToDevice,
                                       s.raw()));
        }
      }
    } catch (std::exception &) {
      cudaFree(dev_);
      cudaFreeHost(host_);
      throw;
    }
  }

  gpu_mat(size_t        height,
          size_t        width,
          value_type    val,
          const stream &s = stream{0})
      : host_{nullptr}
      , dev_{nullptr}
      , height_{height}
      , width_{width} {
    if (this->empty()) {
      return;
    }

    try {
      size_t bytes = this->bytes();
      THROW_IF_ERR(cudaMalloc(&dev_, bytes));

      if (s.is_default() == false) {
        cudaHostAlloc(&host_, bytes, cudaHostAllocDefault);
      }
    } catch (std::exception &e) {
      cudaFree(dev_);
      cudaFreeHost(host_);
      throw;
    }

    detail::fill<<<GET_GRID_DIM(*this), GET_BLOCK_DIM(*this), 0, s.raw()>>>(
        dev_,
        height_,
        width_,
        val);
  }

  gpu_mat(size_t height = 0, size_t width = 0, const stream &s = stream{})
      : host_{nullptr}
      , dev_{nullptr}
      , height_{height}
      , width_{width} {
    if (this->empty()) {
      return;
    }

    try {
      size_t bytes = this->bytes();
      THROW_IF_ERR(cudaMalloc(&dev_, bytes));

      if (s.is_default() == false) {
        cudaHostAlloc(&host_, bytes, cudaHostAllocDefault);
      }
    } catch (std::exception &e) {
      cudaFree(dev_);
      cudaFreeHost(host_);
      throw;
    }
  }

  ~gpu_mat() {
    cudaFreeHost(host_);
    cudaFree(dev_);
  }

  gpu_mat(const gpu_mat &) = delete;
  gpu_mat &operator=(const gpu_mat &) = delete;

  gpu_mat(gpu_mat &&rhs)
      : host_{rhs.host_}
      , dev_{rhs.dev_}
      , height_{rhs.height_}
      , width_{rhs.width_} {
    rhs.dev_ = nullptr;
  }
  gpu_mat &operator=(gpu_mat &&rhs) {
    cudaFree(this->dev_);

    this->host_   = rhs.host_;
    this->dev_    = rhs.dev_;
    this->height_ = rhs.height_;
    this->width_  = rhs.width_;

    rhs.host_ = nullptr;
    rhs.dev_  = nullptr;

    return *this;
  }


  /**\brief download matrix from gpu to cpu memory
   * \param total count of elements for download
   * \param data pointer to allocated memory there we copy. Must be on cpu
   * \note don't use with streams
   */
  void download(void *data, size_t total) {
    THROW_IF_ERR(cudaMemcpy(data,
                            dev_,
                            total * sizeof(value_type),
                            cudaMemcpyDeviceToHost));
  }

  /**\brief download all matrix from gpu to cpu memory
   * \param data pointer to allocated memory that can fit the entire matrix
   * \note don't use with streams
   */
  void download(void *data) {
    this->download(data, this->total());
  }

  /**\brief download matrix from gpu to blocked host memory, you can get
   * access to it by host_ptr
   */
  void download(const stream &s) {
    size_t total = this->total();
    THROW_IF_ERR(cudaMemcpyAsync(host_,
                                 dev_,
                                 total * sizeof(value_type),
                                 cudaMemcpyDeviceToHost,
                                 s.raw()));
  }

  size_t height() const noexcept {
    return height_;
  }

  size_t width() const noexcept {
    return width_;
  }

  size_t total() const noexcept {
    return width_ * height_;
  }

  bool empty() const noexcept {
    return this->total() == 0;
  }

  /**\brief allocated memory size
   */
  size_t bytes() const noexcept {
    return this->total() * sizeof(value_type);
  }


  value_type *device_ptr() noexcept {
    return dev_;
  }

  value_type *host_ptr() noexcept {
    return host_;
  }


private:
  value_type *host_;
  value_type *dev_;
  size_t      height_;
  size_t      width_;
};


/**\note not owns memory of gpu_mat, so gpu_mat must exists all time while
 * you use gpu_map_ptr
 */
template <typename T>
class gpu_mat_ptr {
public:
  using value_type = T;

  gpu_mat_ptr(gpu_mat<value_type> &mat)
      : dev_{mat.device_ptr()}
      , height_{mat.height()}
      , width_{mat.width()} {
  }
  ~gpu_mat_ptr() = default;


  __device__ __host__ size_t height() const noexcept {
    return height_;
  }

  __device__ __host__ size_t width() const noexcept {
    return width_;
  }

  __device__ value_type &operator()(size_t row, size_t col) {
    return dev_[row * width_ + col];
  }

  __device__ value_type operator()(size_t row, size_t col) const {
    return dev_[row * width_ + col];
  }


private:
  value_type *dev_;
  size_t      height_;
  size_t      width_;
};


template <typename T>
gpu_mat_ptr<T> make_gpu_mat_ptr(gpu_mat<T> &mat) {
  return gpu_mat_ptr<T>(mat);
}

template <typename T>
const gpu_mat_ptr<T> make_gpu_mat_ptr(const gpu_mat<T> &mat) {
  return gpu_mat_ptr<T>(const_cast<gpu_mat<T> &>(mat));
}
} // namespace cuda
