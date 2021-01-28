// remap.hpp
/**\file
 */

#include "gpu_mat.hpp"
#include "misc.hpp"


namespace cuda {

namespace detail {
template <typename IOType, typename MapType>
__global__ void remap(const gpu_mat_ptr<IOType>  input,
                      gpu_mat_ptr<IOType>        output,
                      const gpu_mat_ptr<MapType> map_x,
                      const gpu_mat_ptr<MapType> map_y) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(input, row);
  GET_COL_OR_RETURN(input, col);


  MapType x = map_x(row, col);
  MapType y = map_y(row, col);

  if (y < input.height() && y >= 0 && x < input.width() && x >= 0) {
    output(row, col) = input(y, x);
  }
}
} // namespace detail


/**\brief remap input matrix to output matrix by map_y and map_x
 * \param map_x x mapping, matrix must have same size as input matrix
 * \param map_y y mapping, matrix must have same size as input matrix
 * \param s if stream is not default, then remapping will be asynchronous
 * \see opencv::remap
 */
template <typename IOType, typename MapType>
void remap(const gpu_mat<IOType> & input,
           gpu_mat<IOType> &       output,
           const gpu_mat<MapType> &map_x,
           const gpu_mat<MapType> &map_y,
           const stream &          s = stream{0}) {
  ASSERT_ARG(map_x.width() == map_y.width() && map_x.height() == map_y.height(),
             "map_x and map_y has different sizes");
  ASSERT_ARG(input.width() == map_x.width() && input.height() == map_x.height(),
             "map_x/y and input has different sizes");

  if (input.width() != output.width() || input.height() != output.height()) {
    output = gpu_mat<IOType>(input.height(), input.width());
  }

  detail::remap<<<input.height(), input.width(), 0, s.raw()>>>(
      make_gpu_mat_ptr(input),
      make_gpu_mat_ptr(output),
      make_gpu_mat_ptr(map_x),
      make_gpu_mat_ptr(map_y));
}
} // namespace cuda
