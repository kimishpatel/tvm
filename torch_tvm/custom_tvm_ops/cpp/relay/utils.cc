#include "utils.h"

namespace tvm {
namespace relay {

namespace helper {
int32_t get_pack_width(int32_t dim_size) {
  int32_t pack_width = 1;
  int32_t vec_width = 32;
  while (vec_width > 1) {
    if (dim_size % vec_width == 0) {
      return vec_width;
    }
    vec_width /= 2;
  }
  return 1;
}
} // namespace helper
} // namespace relay
} // namespace tvm
