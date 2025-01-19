import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.transformer.attention import MultiHeadAttention
import torch.nn as nn
from super_quantization.super_quantization import QuantizedLayer
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from xy_dataset import XyDataset

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


    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 20,
        mini_batch_size: int = 16,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: optim.Optimizer = optim.AdamW(),
        grad_clamp: float = 1,
    ) -> None:
        """
        Train a model to minimize the loss between predicted and target outputs.
        
        Parameters:
        - X (torch.Tensor): Input tensors.
        - y (torch.Tensor): Target tensors.
        - epochs (int): Number of training epochs.
        - mini_batch_size (int): Batch size for mini-batches.
        - criterion (nn.Module): Loss function.
        - optimizer (optim.Optimizer): Optimizer for gradient updates.
        - grad_clamp (float): Maximum gradient value for clipping.
        """
        self.train()
        dataset = XyDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for mini_batch, target in dataloader:
                optimizer.zero_grad()
                output = self.apply(mini_batch)
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clamp)
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")


    def test_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module = torch.nn.CrossEntropyLoss,
    ) -> float:
        """
        Test a model on a given dataset.

        Parameters:
        - model (nn.Module): The model to train.
        - criterion (nn.Module): Loss function.
        - X (list[torch.Tensor]): Input tensors.
        - y (list[torch.Tensor]): Target tensors.
        """

        self.eval()
        loss = 0
        for mini_batch, target in zip(X, y):
            output = self.apply(mini_batch)
            loss += criterion(output, target)
        loss /= X.shape[0]
        print(f"Score on the whole set, loss: {loss}")
        return loss.item()

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
