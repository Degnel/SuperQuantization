import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.transformer.attention import MultiHeadAttention
import torch.nn as nn
from super_quantization.super_quantization import QuantizedLayer
import torch


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
        vocab_size=None,
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

        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
            self.output_projection.weight = self.embedding.weight
        else:
            self.embedding = None
            self.output_projection = None

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        if self.embedding is not None:
            x = x.to(torch.int32)
            x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        if self.output_projection is not None:
            x = self.output_projection(x)
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
            # self.custom_init(self.fc_1.weight)

        if quantize_fc_2:
            self.fc_2 = QuantizedLayer(d_ff, d_model)
        else:
            self.fc_2 = nn.Linear(d_ff, d_model)
            # self.custom_init(self.fc_2.weight)

        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.previous_weights = None

    def custom_init(self, tensor):
        with torch.no_grad():
            tensor.uniform_(-1, 2)
            tensor.round_()
            tensor.clamp_(-1, 2)

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        # x = F.normalize(x + self.dropout(attn_output), p=2, dim=-1)
        # x = F.normalize(x + attn_output, p=2, dim=-1)

        # Réseau feed-forward
        ff_output = self.fc_2(self.activation(self.fc_1(x)))
        x = self.layer_norm2(x + self.dropout(ff_output))
        # x = F.normalize(x + self.dropout(ff_output), p=2, dim=-1)
        # x = F.normalize(x + ff_output, p=2, dim=-1)
        self.check_weights()
        return x

    def check_weights(self):
        current_weights = {
            name: param.clone() for name, param in self.named_parameters()
        }

        if self.previous_weights is not None:
            for name, param in current_weights.items():
                param = torch.round(torch.clamp(param, -1, 2))
                self.previous_weights[name] = torch.round(
                    torch.clamp(self.previous_weights[name], -1, 2)
                )
                pass
                # assert torch.equal(param, self.previous_weights[name]), f"Les poids pour {name} ont changé."

        self.previous_weights = current_weights
