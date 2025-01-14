import torch.nn as nn
from .gelu import GELU
from super_quantization.super_quantization import QuantizedLayer


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, quantize=True):
        super(PositionwiseFeedForward, self).__init__()
        if quantize:
            self.w_1 = QuantizedLayer(d_model, d_ff)
            self.w_2 = QuantizedLayer(d_ff, d_model)
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
