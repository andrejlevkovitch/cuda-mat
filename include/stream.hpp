// stream.hpp

#pragma once

#include "misc.hpp"
#include <cuda_runtime.h>


namespace cuda {
class stream {
public:
  /**\param create if true, then will be created new cuda stream, otherwise
   * stream not uses
   */
  explicit stream(bool create = true)
      : stream_{0} {
    if (create) {
      THROW_IF_ERR(cudaStreamCreate(&stream_));
    }
  }

  ~stream() {
    if (stream_ != 0) {
      cudaStreamDestroy(stream_);
    }
  }


  stream(const stream &) = delete;
  stream &operator=(const stream &) = delete;


  stream(stream &&rhs)
      : stream_{rhs.stream_} {
  }
  stream &operator=(stream &&rhs) {
    this->stream_ = rhs.stream_;
    rhs.stream_   = 0;
    return *this;
  }


  cudaStream_t raw() const noexcept {
    return stream_;
  }

  bool is_default() const noexcept {
    return stream_ == 0;
  }


  void synchronize() {
    THROW_IF_ERR(cudaStreamSynchronize(stream_));
  }

private:
  cudaStream_t stream_;
};
} // namespace cuda
