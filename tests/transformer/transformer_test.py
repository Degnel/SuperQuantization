from transformer import Transformer
from density.space import ArchitecturalSpace
from density.probabilistic_density import ArchitectureComparator
from torch import optim
from torch import nn
import torch
import torch.nn.functional as F

"""
In this exemple we are comparing 3 transformers with the same general architecture
sq_transformer have all its weigths quantized into int2
transformer have all its weigths but is 16 times narrower to compensate for the float32 precision
full_transformer have both the full size and precision and will be the third party architecture to compare with
"""

d_model = 6
seq_length = 5
n_heads = 1
sq_n_heads = 1
d_ff = 6
sq_d_ff = 6
vocab_size = 64
max_depth = 1
quantize_Q = True
quantize_K = True
quantize_V = True
quantize_fc_1 = False
quantize_fc_2 = False

# Create competing architectures
sq_transformer_params = [
    {
        "d_model": d_model,
        "n_heads": sq_n_heads,
        "d_ff": sq_d_ff,
        "depth": i + 4,
        "quantize_Q": quantize_Q,
        "quantize_K": quantize_K,
        "quantize_V": quantize_V,
        "quantize_fc_1": quantize_fc_1,
        "quantize_fc_2": quantize_fc_2,
        "vocab_size": vocab_size,
    }
    for i in range(max_depth)
]

transformer_params = [
    {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "depth": i + 4,
        "vocab_size": vocab_size,
    }
    for i in range(max_depth)
]

full_transformer_params = [
    {
        "d_model": d_model,
        "n_heads": sq_n_heads,
        "d_ff": sq_d_ff,
        "depth": i + 4,
        "vocab_size": vocab_size,
    }
    for i in range(max_depth)
]


def mesure_information(
    d_model,
    d_ff,
    n_heads,
    depth,
    quantize_Q=False,
    quantize_K=False,
    quantize_V=False,
    quantize_fc_1=False,
    quantize_fc_2=False,
):
    count = 0
    count += bit_diff(quantize_Q) * d_model**2
    count += bit_diff(quantize_K) * d_model**2
    count += bit_diff(quantize_V) * d_model**2
    count *= n_heads
    count += bit_diff(quantize_fc_1) * d_model * d_ff + 32 * d_ff
    count += bit_diff(quantize_fc_2) * d_ff * d_model + 32 * d_model
    count *= depth
    return count


def bit_diff(boolean):
    return 32 * (not boolean) + 2 * boolean


# Create architectural spaces
epoch = [i + 10 for i in range(max_depth)]

sq_transformer_mesurement = [
    mesure_information(
        d_model,
        d_ff,
        sq_n_heads,
        i + 4,
        quantize_Q,
        quantize_K,
        quantize_V,
        quantize_fc_1,
        quantize_fc_2,
    )
    for i in range(max_depth)
]

sq_transformer_space = ArchitecturalSpace(
    (seq_length,),
    "Super Quantized Transformer",
    Transformer,
    sq_transformer_params,
    lr=0.001,
    mini_batch_size=16,
    epoch=epoch,
    mesurement=sq_transformer_mesurement,
    optimizer=optim.AdamW,
)

transformer_mesurement = [
    mesure_information(d_model, d_ff, n_heads, i + 4) for i in range(max_depth)
]

transformer_space = ArchitecturalSpace(
    (seq_length,),
    "Transformer",
    Transformer,
    transformer_params,
    epoch=epoch,
    mesurement=transformer_mesurement,
)

full_transformer_mesurement = [
    mesure_information(d_model, d_ff, sq_n_heads, i + 4) for i in range(max_depth)
]

full_transformer_space = ArchitecturalSpace(
    (seq_length,),
    "Full Transformer",
    Transformer,
    full_transformer_params,
    epoch=epoch,
    mesurement=full_transformer_mesurement,
)

# Create comparator
# comparator = ArchitectureComparator(
#     sq_transformer_space, transformer_space, full_transformer_space
# )

# comparator = ArchitectureComparator(
#     transformer_space, sq_transformer_space, full_transformer_space
# )


class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
    
    def forward(self, logits_pred, logits_target):
        prob_pred = F.softmax(logits_pred, dim=-1)
        prob_target = F.softmax(logits_target, dim=-1)
        
        loss = -torch.sum(prob_target * torch.log(prob_pred + 1e-8), dim=-1)

        return loss.mean()

comparator = ArchitectureComparator(
    sq_transformer_space,
    transformer_space,
    criterion=CustomCrossEntropyLoss(),
    law=torch.distributions.Categorical(probs=torch.ones(vocab_size)),
)

res = comparator.compare()
print(res)
comparator.plot("min")