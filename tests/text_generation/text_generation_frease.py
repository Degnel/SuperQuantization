import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.transformer import Transformer
from super_quantization.super_quantizer import SuperQuantizer
from tests.text_generation.preprocessing import get_data
from torch.utils.data import DataLoader
import torch
from frEase.recipes import ProgressiveRecipes
from frEase.trainer import ProgressiveTrainer
from torch import optim, nn

seq_length = 5
vocab_size = 10000
train_batch_count = 2000
test_batch_count = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sq = SuperQuantizer()

sq_dim = 32
# On entraine notre transformer
sq_model = Transformer(
    d_model=33,
    n_heads=2,
    d_ff=32,
    depth=4,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=True,
    quantize_fc_2=True,
    vocab_size=vocab_size,
    max_context_size=seq_length,
).to(device)

# sq_model.train_model(
#     train_dataloader, epochs=200, lr=0.001
# )

# sq_model.test_model(validation_dataloader)
sq_mesure = sq.mesure(sq_model)
print("Total bits of information in super quantized model: ", sq_mesure)
print("Active total bits of information in super quantized model: ", sq_mesure - 32 * sq_dim * (2*vocab_size - seq_length))

train_dataset, validation_dataset, vocab = get_data(
    seq_length, vocab_size, train_batch_count, test_batch_count
)
train_dataloader = DataLoader(train_dataset, batch_size=16)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)

sq_recipe = ProgressiveRecipes(sq_model)
sq_recipe.base_recipe(epochs=40, iterations=10, global_trainning=0, scaling_factor=1, constructive=True)
trainer = ProgressiveTrainer(sq_recipe)
trainer.train(
    train_dataloader, optim.AdamW, nn.CrossEntropyLoss(), validation_dataloader
)