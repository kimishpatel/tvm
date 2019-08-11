#pragma  once

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir.h>

#include <dlpack/dlpack.h>

#include <memory>

namespace torch_tvm {
namespace utils {

struct DLManagedTensorDeleter {
  void operator()(DLManagedTensor* manager_ctx) {
    if (manager_ctx == nullptr)
      return;
    auto dl_tensor = manager_ctx->dl_tensor;
    if (dl_tensor.data) {
      delete dl_tensor.data;
      // Implicit assumption is made that data, shape and strides are
      // together either all nullptr or not.
      delete dl_tensor.shape;
      delete dl_tensor.strides;
    }
    delete manager_ctx;
  }
};

bool is_aligned(void* data_ptr, std::uintptr_t alignment_in_bytes);

DLManagedTensor* alloc_and_copy_data(const at::Tensor& tensor);
using DLManagedTensorPtr = std::unique_ptr<DLManagedTensor,
      DLManagedTensorDeleter>;

} // utils
} // torch_tvm
