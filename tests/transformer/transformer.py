import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.transformer.attention import MultiHeadAttention
import torch.nn as nn
from super_quantization.super_quantization import QuantizedLayer


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        depth,
        dropout=0.1,
        quantize_Q=False,
        quantize_K=False,
        quantize_V=False,
        quantize_fc_1=False,
        quantize_fc_2=False,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Liste des couches de l'encodeur
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    quantize_Q,
                    quantize_K,
                    quantize_V,
                    quantize_fc_1,
                    quantize_fc_2,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        quantize_Q=False,
        quantize_K=False,
        quantize_V=False,
        quantize_fc_1=False,
        quantize_fc_2=False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, quantize_Q, quantize_K, quantize_V
        )

        if quantize_fc_1:
            self.fc_1 = QuantizedLayer(d_model, d_ff)
        else:
            self.fc_1 = nn.Linear(d_model, d_ff)

        if quantize_fc_2:
            self.fc_2 = QuantizedLayer(d_ff, d_model)
        else:
            self.fc_2 = nn.Linear(d_ff, d_model)

        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Réseau feed-forward
        ff_output = self.fc_2(self.activation(self.fc_1(x)))
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x
