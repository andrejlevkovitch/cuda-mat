// misc.hpp

#pragma once

#include "exception.hpp"
#include <stdexcept>


#define THROW_IF_ERR(op)                                                       \
  {                                                                            \
    cudaError_t cuda_operation_status = op;                                    \
    if (cuda_operation_status != cudaError::cudaSuccess) {                     \
      throw cuda::exception{cudaGetErrorString(cuda_operation_status)};        \
    }                                                                          \
  }


#define ASSERT_ARG(val, msg)                                                   \
  if ((val) == false) {                                                        \
    throw std::invalid_argument{msg};                                          \
  }
