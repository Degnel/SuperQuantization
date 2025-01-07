import torch
import torch.nn as nn
from super_quantization import QuantizedLayer

class QuantizedNet(nn.Module):
    def __init__(self, depth, input_dim=20, hidden_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [QuantizedLayer(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([-1, 0, 1, 2], dtype=torch.float32)[
                    torch.randint(0, 4, layer.weight.size(), dtype=torch.long)
                ])

    def forward(self, x):
        skip = x
        for layer in self.layers:
            out = self.relu(layer(skip))
            skip = skip + out
            skip = skip / torch.norm(skip, p=2, dim=1, keepdim=True)
        return skip