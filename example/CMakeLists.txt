# cmake for examples

macro(add_example EXAMPLE_NAME EXAMPLE_FILE)
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${EXAMPLE_NAME} PRIVATE
    ${PROJECT_NAME}
    )
endmacro()

add_example(info        info.cu)
add_example(simple      simple.cu)
add_example(remap       remap.cu)
add_example(remap_async remap_async.cu)
add_example(warp_affine warp_affine.cu)


find_package(OpenCV COMPONENTS core imgcodecs REQUIRED)

add_executable(bend bend.cu)
target_include_directories(bend PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(bend PRIVATE
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  )


add_executable(is_dublicates is_dublicates.cu)
target_include_directories(is_dublicates PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(is_dublicates PRIVATE
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  )


add_executable(blend blend.cu)
target_include_directories(blend PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(blend PRIVATE
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  )
