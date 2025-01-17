import torch
import torch.nn as nn
from math import sqrt


class QuantizedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, lr_scale=1) -> None:
        super().__init__()
        std = sqrt(2 / input_dim)
        self.weight = nn.Parameter(
            # (torch.randint(0, 4, (input_dim, output_dim)) - 1).float()
            # torch.zeros(input_dim, output_dim, dtype=torch.float32),
            torch.randn(input_dim, output_dim, dtype=torch.float32)
            * std
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim, dtype=torch.float32))
        else:
            self.bias = None
        self.lr_scale = lr_scale

    def forward(self, x) -> torch.Tensor:
        if self.bias is None:
            return DiscreteMatrixMultiply.apply(x, self.weight, self.lr_scale)
        else:
            return (
                DiscreteMatrixMultiply.apply(x, self.weight, self.lr_scale) + self.bias
            )


class DiscreteMatrixMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix, weight_matrix, lr_scale=10000) -> torch.Tensor:
        ctx.save_for_backward(input_matrix, weight_matrix)
        ctx.lr_scale = lr_scale
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        r = input_matrix @ quantized_weight
        # print("input_matrix: ", input_matrix)
        # print("quantized_weight: ", quantized_weight)
        # print("r: ", r)
        return r

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor]:
        input_matrix, weight_matrix = ctx.saved_tensors
        lr_scale = ctx.lr_scale
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        grad_input = grad_output @ quantized_weight.transpose(-1, -2)
        # grad_input = grad_output @ weight_matrix.transpose(-1, -2)
        grad_weight = input_matrix.transpose(-1, -2) @ grad_output
        grad_weight = torch.clamp(lr_scale * grad_weight, -1, 1)
        # grad_weight = lr_scale*grad_weight
        # print((torch.abs(lr_scale*grad_weight)>1).float().mean().item())
        # mean = (torch.abs(lr_scale*grad_weight)>1).float().mean().item()
        # if mean != 0:
        #     print('Success!!', mean)

        return grad_input, grad_weight, None
