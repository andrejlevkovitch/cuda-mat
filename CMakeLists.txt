# cmake

cmake_minimum_required(VERSION 3.14)

project(cuda-mat LANGUAGES CUDA CXX)

include(cmake/build.cmake)
include(cmake/doxygen.cmake)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  include
  )
#set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)


add_executable(simple example/simple.cu)
target_link_libraries(simple PRIVATE
  ${PROJECT_NAME}
  )


add_executable(remap example/remap.cu)
target_link_libraries(remap PRIVATE
  ${PROJECT_NAME}
  )

add_executable(remap_async example/remap_async.cu)
target_link_libraries(remap_async PRIVATE
  ${PROJECT_NAME}
  )

add_executable(info example/info.cu)
