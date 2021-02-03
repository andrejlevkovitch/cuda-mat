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
                      const gpu_mat_ptr<MapType> map,
                      IOType                     border_value) {
  unsigned int row = 0;
  unsigned int col = 0;
  GET_ROW_OR_RETURN(output, row);
  GET_COL_OR_RETURN(output, col);


  MapType map_vec = map(row, col);

  if (map_vec.y < input.height() && map_vec.y >= 0 &&
      map_vec.x < input.width() && map_vec.x >= 0) {
    output(row, col) = input(map_vec.y, map_vec.x);
  } else {
    output(row, col) = border_value;
  }
}
} // namespace detail


/**\brief remap input matrix to output matrix by map_y and map_x
 * \param map mapping, matrix must have same size as input matrix. Values must
 * be 2d vectors
 * \param s if stream is not default, then remapping will be asynchronous
 * \param interpolation not used, reserver for future, now uses only nearest
 * mode
 * \param border_mode not used, used only constant border mode
 * \param border_value value for elements, that not present in input
 * \note by default border_value uses default konstructor, so it don't garantie
 * that border_value will a zero
 * \see opencv::remap
 * \warning usage one matrix for input and output produce undefined behaviour
 */
template <typename IOType, typename MapType>
void remap(const gpu_mat<IOType> & input,
           gpu_mat<IOType> &       output,
           const gpu_mat<MapType> &map,
           int                     interpolation = 0,
           int                     border_mode   = 0,
           IOType                  border_value  = {},
           const stream &          s             = stream{0}) {
  if (map.width() != output.width() || map.height() != output.height()) {
    output = gpu_mat<IOType>(map.height(), map.width(), s);
  }

  if (output.empty()) {
    return;
  }


  detail::remap<<<GET_GRID_DIM(output), GET_BLOCK_DIM(output), 0, s.raw()>>>(
      make_gpu_mat_ptr(input),
      make_gpu_mat_ptr(output),
      make_gpu_mat_ptr(map),
      border_value);
}
} // namespace cuda
