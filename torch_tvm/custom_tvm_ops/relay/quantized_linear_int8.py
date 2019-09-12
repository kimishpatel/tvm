from __future__ import absolute_import

import topi
from tvm.relay.op import op as reg
from tvm.relay.op.op import OpPattern, schedule_injective
from topi.util import get_const_int
from tvm import autotvm
import tvm

from torch_tvm.custom_tvm_ops.topi import quantized_linear_int8

@reg.register_compute("nn.quantize_data_mm_dequantize")
def compute_quantized_mm_dequantize(attrs, inputs, out_type, target):
    data = inputs[0]
    weight = inputs[1]
    weight_acc = inputs[2]
    data_acc = inputs[3]
    data_scale = inputs[4]
    data_zero_point = inputs[5]
    weight_scale = attrs.w_scale
    weight_zero_point = attrs.w_zp
    out = quantized_linear_int8.quantized_mm_dequantize(data, weight, \
            weight_acc, data_acc, data_scale, data_zero_point, weight_scale, \
            weight_zero_point, out_type.dtype)
    return [out]


@reg.register_schedule("nn.quantize_data_mm_dequantize")
def schedule_quantized_mm_dequantize(attrs, outs, target):
    with target:
        return quantized_linear_int8.schedule_quantized_mm_dequantize(outs)


