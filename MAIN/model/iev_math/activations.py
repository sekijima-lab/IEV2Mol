"""Optional scalar-libm compatibility for the validated historical CPU build.
The first-order derivatives follow PyTorch 1.13.1's activation expressions.
No old PyTorch library is loaded. Higher-order gradients are unsupported.
"""
import numpy as np
import torch
from torch.autograd.function import once_differentiable


def _libm(x, operation):
    if x.device.type != 'cpu' or x.dtype != torch.float32:
        raise ValueError('Old-compatible math requires CPU float32; use --no-old-compatible for standard math')
    from . import _legacy_math
    source = np.ascontiguousarray(x.detach().numpy())
    output = np.empty_like(source)
    getattr(_legacy_math, operation)(source, output)
    return torch.from_numpy(output).reshape(x.shape)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind):
        ctx.kind = kind
        if kind == 'exp':
            output = _libm(x, 'exp')
            ctx.save_for_backward(output)
            return output
        if kind == 'selu':
            negcoef = float(np.float32(np.float32(1.6732632423543772)*np.float32(1.0507009873554805)))
            output = torch.where(x > 0, x*1.0507009873554805, (_libm(x, 'exp')-1)*negcoef)
            ctx.save_for_backward(x)
            ctx.negcoef = negcoef
            return output
        if kind == 'tanh':
            output = _libm(x, 'tanh')
            ctx.save_for_backward(output)
            return output
        exponential = _libm(-x, 'exp')
        sigmoid = 1 / (1 + exponential)
        ctx.save_for_backward(x, sigmoid)
        return sigmoid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if ctx.kind == 'exp':
            output, = ctx.saved_tensors
            return grad*output, None
        if ctx.kind == 'selu':
            x, = ctx.saved_tensors
            return torch.where(x > 0, grad*1.0507009873554805, grad*ctx.negcoef*_libm(x,'exp')), None
        if ctx.kind == 'tanh':
            output, = ctx.saved_tensors
            return grad * (1 - output * output), None
        x, sigmoid = ctx.saved_tensors
        return grad * (1 - sigmoid) * sigmoid, None



def exp(x):
    return _Activation.apply(x, 'exp')

def selu(x):
    return _Activation.apply(x, 'selu')
