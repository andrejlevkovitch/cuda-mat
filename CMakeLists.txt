# cmake

cmake_minimum_required(VERSION 3.14)

project(cuda-mat LANGUAGES CUDA CXX)

include(cmake/build.cmake)
include(cmake/doxygen.cmake)

add_subdirectory(third-party)


add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  include
  )


add_subdirectory(example)
add_subdirectory(test)
