from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity

from topi.x86.util import get_fp32_len
from topi.x86.dense import _default_dense_pack_config
from topi import generic, tag, nn
from topi.util import traverse_inline, get_const_tuple


@autotvm.register_topi_compute(nn.dense, "cpu", "direct", override=True)
def _declaration_custom_dense(cfg, data, packed_weight, bias, out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    pack_outer, _, pack_width = get_const_tuple(packed_weight.shape)
    out_dim = pack_outer * pack_width
    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=3)
    cfg.define_split("tile_x", out_dim, num_outputs=3)
    cfg.define_split("tile_k", in_dim, num_outputs=2)
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, batch, out_dim, in_dim)

    k = tvm.reduce_axis((0, in_dim), name="k")
    C = tvm.compute((batch, out_dim),
                    lambda y, x: tvm.sum(
                        data[y, k].astype(out_dtype) * \
                        packed_weight[x // pack_width, k, x % pack_width].astype(out_dtype),
                        axis=k),
                    tag="custom_dense_pack")
    if bias is not None:
        C = tvm.compute((batch, out_dim), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)
    return C


@autotvm.register_topi_schedule(generic.schedule_dense, "cpu", "direct", override=True)
def _schedule_custom_dense_pack(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _schedule_custom_dense_pack_template(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_custom_dense_pack_template(cfg, s, C):
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis

    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    return s


def dense_weight_pack(weight, pack_width):
    N, K = get_const_tuple(weight.shape)

    packw_shape = (N // pack_width, K, pack_width)
    C = tvm.compute(packw_shape, \
            lambda z, y, x: weight[z * pack_width+ x, y], name="packed_weight")
    return C


def schedule_dense_weight_pack(outs):
    s = tvm.create_schedule([x.op for x in outs])
    packedB = outs[0].op
    z, y, x = s[packedB].op.axis
    s[packedB].reorder(z, x, y)
    s[packedB].parallel(z)
    s[packedB].vectorize(y)
    return s
