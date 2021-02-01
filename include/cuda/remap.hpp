// remap.hpp
/**\file
 */

#include "cuda/gpu_mat.hpp"
#include "cuda/misc.hpp"


namespace cuda {

namespace detail {
template <typename IOType, typename MapType>
__global__ void remap(const gpu_mat_ptr<IOType>  input,
                      gpu_mat_ptr<IOType>        output,
                      const gpu_mat_ptr<MapType> map_x,
                      const gpu_mat_ptr<MapType> map_y,
                      IOType                     border_value) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(input, row);
  GET_COL_OR_RETURN(input, col);


  MapType x = map_x(row, col);
  MapType y = map_y(row, col);

  if (y < input.height() && y >= 0 && x < input.width() && x >= 0) {
    output(row, col) = input(y, x);
  } else {
    output(row, col) = border_value;
  }
}
} // namespace detail


/**\brief remap input matrix to output matrix by map_y and map_x
 * \param map_x x mapping, matrix must have same size as input matrix
 * \param map_y y mapping, matrix must have same size as input matrix
 * \param s if stream is not default, then remapping will be asynchronous
 * \param interpolation not used, reserver for future, now uses only nearest
 * mode
 * \param border_mode not used, used only constant border mode
 * \param border_value value for elements, that not present in input
 * \see opencv::remap
 */
template <typename IOType, typename MapType>
void remap(const gpu_mat<IOType> & input,
           gpu_mat<IOType> &       output,
           const gpu_mat<MapType> &map_x,
           const gpu_mat<MapType> &map_y,
           int                     interpolation = 0,
           int                     border_mode   = 0,
           IOType                  border_value  = 0,
           const stream &          s             = stream{0}) {
  ASSERT_ARG(map_x.width() == map_y.width() && map_x.height() == map_y.height(),
             "map_x and map_y has different sizes");
  ASSERT_ARG(input.width() == map_x.width() && input.height() == map_x.height(),
             "map_x/y and input has different sizes");

  if (input.width() != output.width() || input.height() != output.height()) {
    output = gpu_mat<IOType>(input.height(), input.width(), s);
  }

  if (input.empty()) {
    return;
  }

  detail::remap<<<GET_GRID_DIM(input), GET_BLOCK_DIM(input), 0, s.raw()>>>(
      make_gpu_mat_ptr(input),
      make_gpu_mat_ptr(output),
      make_gpu_mat_ptr(map_x),
      make_gpu_mat_ptr(map_y),
      border_value);
}
} // namespace cuda
