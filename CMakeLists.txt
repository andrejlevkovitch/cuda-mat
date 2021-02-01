# cmake

cmake_minimum_required(VERSION 3.14)

project(cuda-mat LANGUAGES CUDA CXX)

include(cmake/build.cmake)
include(cmake/doxygen.cmake)

add_subdirectory(third-party)


# XXX remove in future
# need for inversing matricies
find_package(OpenCV COMPONENTS core REQUIRED)


add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  include
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} INTERFACE
  ${OpenCV_LIBS}
  )


add_subdirectory(example)
add_subdirectory(test)
