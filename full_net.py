import torch
import torch.nn as nn
import torch.nn.init as init

class FullPrecisionNet(nn.Module):
    def __init__(self, depth, input_dim=20, hidden_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(depth)]
        )
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        skip = x
        for layer in self.layers:
            out = self.relu(layer(skip))
            skip = skip + out
            skip = skip / torch.norm(skip, p=2, dim=1, keepdim=True)
        return skip