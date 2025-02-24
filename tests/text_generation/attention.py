import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from super_quantization.super_quantization import QuantizedLayer

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        quantize_Q=False,
        quantize_K=False,
        quantize_V=False,
        quantize_O=False,
        lora_ratio=4,
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.lora_dim = int(d_model / lora_ratio)

        # Projection pour Q
        if quantize_Q:
            self.Q = QuantizedLayer(d_model, self.lora_dim * n_heads, False)
        else:
            self.Q = nn.Linear(d_model, self.lora_dim * n_heads, False)

        # Projection pour K
        if quantize_K:
            self.K = QuantizedLayer(d_model, self.lora_dim * n_heads, False)
        else:
            self.K = nn.Linear(d_model, self.lora_dim * n_heads, False)

        # Projection pour V : on projette en (n_heads * lora_dim)
        if quantize_V:
            self.V = QuantizedLayer(d_model, self.lora_dim * n_heads, False)
        else:
            self.V = nn.Linear(d_model, self.lora_dim * n_heads, False)

        # Couche de sortie unique pour combiner les têtes.
        # L'entrée est de dimension (n_heads * lora_dim) et la sortie de dimension d_model.
        if quantize_O:
            self.O = QuantizedLayer(self.n_heads * self.lora_dim, d_model, False)
        else:
            self.O = nn.Linear(self.n_heads * self.lora_dim, d_model, False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        mask : Tensor de taille (batch_size, 1, seq_len, seq_len) ou (batch_size*n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Calcul de Q, K et V
        q, k, v = self.Q(x), self.K(x), self.V(x)

        # Reshape pour séparer les têtes: (batch_size*n_heads, seq_len, lora_dim)
        query = self._reshape_to_batches(q, self.lora_dim)
        key = self._reshape_to_batches(k, self.lora_dim)
        value = self._reshape_to_batches(v, self.lora_dim)

        # Calcul des scores d'attention
        dk = self.lora_dim
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(scores, dim=-1)

        # Produit pondéré : (batch_size*n_heads, seq_len, lora_dim)
        y = attention.matmul(value)

        # Reforme en (batch_size, n_heads, seq_len, lora_dim)
        y = y.reshape(batch_size, self.n_heads, seq_len, self.lora_dim)
        # Permute pour obtenir (batch_size, seq_len, n_heads, lora_dim) puis concatène les têtes
        y = y.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.lora_dim)

        # Application de la projection de sortie unique
        output = self.O(y)  # (batch_size, seq_len, d_model)
        return output

    def _reshape_to_batches(
        self,
        x: torch.Tensor,
        last_dim: int,
    ) -> torch.Tensor:
        """
        x : Tensor de taille (batch_size, seq_len, n_heads * dim)
        Retourne : Tensor de taille (batch_size*n_heads, seq_len, dim)
        """
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.n_heads, last_dim)
             .permute(0, 2, 1, 3)
             .reshape(batch_size * self.n_heads, seq_len, last_dim)
        )