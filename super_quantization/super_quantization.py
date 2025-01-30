import torch
import torch.nn as nn
from math import sqrt


def clamp01(tensor):
    return torch.where(tensor < 0.5, 0.0, 1.0)


def clamp_11(tensor):
    return torch.sign(tensor)


def clamp_2_112(tensor):
    abs = torch.abs(tensor)
    sign = torch.sign(tensor)
    return sign * torch.where(abs < 1.5, 1, 2)


def clamp_2_101(tensor):
    return torch.clamp(torch.round(tensor), -2, 1)


def clamp_1012(tensor):
    return torch.clamp(torch.round(tensor), -1, 2)


CLAMPING_FUNCTIONS = {
    "01": clamp01,
    "_11": clamp_11,
    "_2_112": clamp_2_112,
    "_2_101": clamp_2_101,
    "_1012": clamp_1012,
}


class QuantizedLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, lr_scale=1, quantize_mode="_11"
    ) -> None:
        super().__init__()
        std = sqrt(2 / in_features)
        self.weight = nn.Parameter(
            # (torch.randint(0, 4, (in_features, out_features)) - 1).float()
            # torch.zeros(in_features, out_features, d(type=torch.float32),
            torch.randn(in_features, out_features, dtype=torch.float32)
            * std
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        self.lr_scale = lr_scale
        self.quantize_mode = quantize_mode

    def __repr__(self):
        return f"QuantizedLayer(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, lr_scale={self.lr_scale}, quantize_mode='{self.quantize_mode}')"

    def forward(self, x) -> torch.Tensor:
        clamping_fn = CLAMPING_FUNCTIONS.get(self.quantize_mode, clamp01)
        if self.bias is None:
            return DiscreteMatrixMultiply.apply(
                x, self.weight, self.lr_scale, clamping_fn
            )
        else:
            return (
                DiscreteMatrixMultiply.apply(x, self.weight, self.lr_scale, clamping_fn)
                + self.bias
            )


class DiscreteMatrixMultiply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_matrix, weight_matrix, lr_scale=1, clamping_fn=clamp_1012
    ) -> torch.Tensor:
        ctx.save_for_backward(input_matrix, weight_matrix)
        ctx.lr_scale = lr_scale
        ctx.clamping_fn = clamping_fn
        quantized_weight = clamping_fn(weight_matrix)
        r = input_matrix @ quantized_weight
        # print("input_matrix: ", input_matrix)
        # print("quantized_weight: ", quantized_weight)
        # print("r: ", r)
        return r

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor | None]:
        input_matrix, weight_matrix = ctx.saved_tensors
        lr_scale = ctx.lr_scale
        clamping_fn = ctx.clamping_fn
        quantized_weight = clamping_fn(weight_matrix)
        grad_input = grad_output @ quantized_weight.transpose(-1, -2)
        # grad_input = grad_output @ weight_matrix.transpose(-1, -2)
        grad_weight = input_matrix.transpose(-1, -2) @ grad_output
        grad_weight = torch.clamp(lr_scale * grad_weight, -1, 1)
        # grad_weight = lr_scale*grad_weight
        # print((torch.abs(lr_scale*grad_weight)>1).float().mean().item())
        # mean = (torch.abs(lr_scale*grad_weight)>1).float().mean().item()
        # if mean != 0:
        #     print('Success!!', mean)

        return grad_input, grad_weight, None, None
