import torch
import torch.nn as nn
from super_quantization import QuantizedLayer

class MNISTQuantizedNet(nn.Module):
    def __init__(self, depth, input_dim=20, hidden_dim=20, output_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList(
            [QuantizedLayer(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.in_layer = QuantizedLayer(input_dim, hidden_dim)
        self.out_layer = QuantizedLayer(hidden_dim, output_dim)
        # self.layers = nn.ModuleList(
        #     [nn.Linear(hidden_dim, hidden_dim, False) for _ in range(depth)]
        # )
        # self.in_layer = nn.Linear(input_dim, hidden_dim, False)
        # self.out_layer = nn.Linear(hidden_dim, output_dim, False)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([-1, 0, 1, 2], dtype=torch.float32)[
                    torch.randint(0, 4, layer.weight.size(), dtype=torch.long)
                ])

    def forward(self, x):
        skip = self.in_layer(x)
        for layer in self.layers:
            out = self.relu(layer(skip))
            skip = skip + out
            skip = skip / torch.norm(skip, p=2, dim=1, keepdim=True)
        y = self.out_layer(skip)
        return y