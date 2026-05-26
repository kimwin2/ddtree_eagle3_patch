import torch


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        deriv = (x > -1) & (x < 1)
        return grad_output * deriv

STEBinary = _STEBinary.apply
