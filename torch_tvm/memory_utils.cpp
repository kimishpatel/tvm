#include "memory_utils.h"

namespace torch_tvm {
namespace utils {

bool is_aligned(void* data_ptr, std::uintptr_t alignment_in_bytes) {
  auto mask = alignment_in_bytes - 1;
  AT_CHECK((alignment_in_bytes & mask) == 0);
  return (reinterpret_cast<std::uintptr_t>(data_ptr) & mask) == 0;
}

DLManagedTensor* alloc_and_copy_data(const at::Tensor& tensor) {
  DLManagedTensor* dl_managed_tensor = new DLManagedTensor();
  // managed_tensor_deleter is supplied to unique_ptr as a deleter
  // of this managed memory. Thus setting deleter to nullptr;
  dl_managed_tensor->deleter = nullptr;
  dl_managed_tensor->manager_ctx = dl_managed_tensor;
  auto& dl_tensor = dl_managed_tensor->dl_tensor;

  auto num_dims = tensor.dim();
  dl_tensor.ndim = num_dims;
  dl_tensor.dtype = at::getDLDataType(tensor);
  int64_t device_id = 0;
  dl_tensor.ctx = getDLContext(tensor, device_id);
  dl_tensor.shape = dl_tensor.strides = nullptr;
  dl_tensor.data = nullptr;
  dl_tensor.shape = new int64_t[num_dims];
  dl_tensor.strides = new int64_t[num_dims];
  AT_CHECK(dl_tensor.shape != nullptr && dl_tensor.strides != nullptr,
      "Memory allocation failed for DLTensor shape and strides"
      "by ManagedTensors.");

  auto tensor_sizes = tensor.sizes();
  auto tensor_strides = tensor.strides();
  for (int64_t i = 0; i < num_dims; ++i) {
    dl_tensor.shape[i] = tensor_sizes[i];
    dl_tensor.strides[i] = tensor_strides[i];
  }
  dl_tensor.data = aligned_alloc(64,
      tensor.nbytes());
  AT_CHECK(dl_tensor.data != nullptr,
      "Memory allocation failed for DLTensor data by ManagedTensors.");

  std::memcpy (dl_tensor.data, tensor.data_ptr(), tensor.nbytes());
  dl_tensor.byte_offset = 0;

  return dl_managed_tensor;
}

} // utils
} // torch_tvm
