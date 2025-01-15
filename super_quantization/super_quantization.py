import torch
import torch.nn as nn
from math import sqrt


class QuantizedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True) -> None:
        super().__init__()
        std = sqrt(2 / input_dim)
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim, dtype=torch.float32) * std
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x) -> torch.Tensor:
        if self.bias is None:
            return DiscreteMatrixMultiply.apply(x, self.weight)
        else:
            return DiscreteMatrixMultiply.apply(x, self.weight) + self.bias


class DiscreteMatrixMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix, weight_matrix) -> torch.Tensor:
        ctx.save_for_backward(input_matrix, weight_matrix)
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        r = input_matrix @ quantized_weight
        # print("input_matrix: ", input_matrix)
        # print("quantized_weight: ", quantized_weight)
        # print("r: ", r)
        return r

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor]:
        input_matrix, weight_matrix = ctx.saved_tensors
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        grad_input = grad_output @ quantized_weight.transpose(-1, -2)
        # grad_input = grad_output @ weight_matrix.t()
        grad_weight = input_matrix.transpose(-1, -2) @ grad_output
        return grad_input, grad_weight
