# Je veux faire des DNN avec mon produit de matrice quantisé
# Le but est d'imiter l'autre avec le même nombre d'information
# Dans un premier temps, je crée un réseau aléatoire avec une succession de couches Lin, ReLU
# Puis je crée un réseau quantisé avec la même quantité d'information
# On rajoutera des skip connections afin de mieux faire passer les gradients
# Le but d'un réseau est d'essayer d'imiter l'autre
# On fera des matrices de dimension (20, 20) et on aura une profondeur variable pour matcher le nombre de paramètres de l'autre réseau
# Si le réseau full a une précision en float32 (32 bits d'info par param), alors le réseau quantisé à toujours 2 bits par paramtères
# Ainsi le réseau quantisé aura le droit à des profondeurs 16 fois plus grandes que celles du réseau full
# On essaie donc avec une profondeur de 16, 32, 48, 64, 80 et 96 pour le réseau quantisé (équivalent à 1, 2, 3, 4, 5 et 6 pour le réseau full)
# On génère donc une première fois un réseau full avec des paramètres aléatoires puis c'est au petit réseau quantisé d'apprendre
# Et vis-versa
# On trace à la fin un joli graphique, montrant 2 courbes montrant l'évolution des MSE en fonction de la profondeur du réseau full   

import torch
import torch.nn as nn

class QuantizedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return DiscreteMatrixMultiply.apply(x, self.weight)
        else:
            return DiscreteMatrixMultiply.apply(x, self.weight) + self.bias

class DiscreteMatrixMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix, weight_matrix):
        ctx.save_for_backward(input_matrix, weight_matrix)
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        r = input_matrix @ quantized_weight
        # print("input_matrix: ", input_matrix)
        # print("quantized_weight: ", quantized_weight)
        # print("r: ", r)
        return r

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix, weight_matrix = ctx.saved_tensors
        quantized_weight = torch.round(torch.clamp(weight_matrix, -1, 2))
        grad_input = grad_output @ quantized_weight.t()
        # grad_input = grad_output @ weight_matrix.t()
        grad_weight = input_matrix.t() @ grad_output
        return grad_input, grad_weight