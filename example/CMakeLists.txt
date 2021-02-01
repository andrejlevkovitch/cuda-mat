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
