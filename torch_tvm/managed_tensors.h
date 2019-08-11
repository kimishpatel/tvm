#pragma once

#include <memory>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <tvm/runtime/device_api.h>

#include <torch/csrc/jit/ir.h>
#include <ATen/DLConvertor.h>

#include "memory_utils.h"

namespace torch_tvm {
namespace {
}

using torch::jit::Value;

using torch_tvm::utils::DLManagedTensorPtr;

class ManagedTensors {
  public:
    ManagedTensors() = default;
    bool insert_value(const Value* val) {
      AT_CHECK(managed_tensors.find(val) == managed_tensors.end(),
          "Attempt to insert Value* when the Value"
          "exists in the ManagedTensors map."
          "You can use value_exist(val) to check if value"
          "already exists.");
      managed_tensors[val] = nullptr;
    }

    void reset_value(const Value* val) {
      AT_CHECK(managed_tensors.find(val) != managed_tensors.end(),
          "Value must have been created inside the managed tensors"
          " in order to reset it.");

      managed_tensors[val] = nullptr;
    }

    bool value_exists(const Value* val) {
      return managed_tensors.find(val) != managed_tensors.end();
    }

    bool tensor_for_value_exists(const Value* val) {
      return managed_tensors.find(val) != managed_tensors.end() &&
        managed_tensors[val] != nullptr;
    }

    void allocate_like_and_map_value(const Value* val, const at::Tensor& tensor) {

      AT_CHECK(managed_tensors.find(val) != managed_tensors.end(),
          "Value must have been created inside the managed tensors"
          " container during graph lowering.");

      auto managed_tensor_ptr = torch_tvm::utils::
        alloc_and_copy_data(tensor);
      managed_tensors[val] = DLManagedTensorPtr(managed_tensor_ptr);
    }

  protected:
    std::unordered_map<const Value*, DLManagedTensorPtr> managed_tensors;
};

class ManagedParamTensors : public ManagedTensors {
  public:
    ManagedParamTensors() = default;
    DLManagedTensor* get_tensor_like_for_value(const Value* val,
        const at::Tensor& tensor) {
      if (!ManagedTensors::tensor_for_value_exists(val)) {
        ManagedTensors::allocate_like_and_map_value(val, tensor);
      }
      return managed_tensors[val].get();
    }
};
} // torch_tvm
