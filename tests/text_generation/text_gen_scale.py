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
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sq = SuperQuantizer()

train_dataset, validation_dataset, vocab = get_data(
    seq_length, vocab_size, train_batch_count, test_batch_count
)

depths = [2, 4, 8, 16]

def compare_models():
    for depth in depths:
        model = Transformer(
            d_model=128*depth,
            n_heads=depth,
            d_ff=512*depth,
            depth=depth,
            vocab_size=vocab_size,
            max_context_size=seq_length,
        ).to(device)
        layers_quant_type = {"K": "01", "Q": "01"}
        sq_model = sq.quantize(sq_model, layers_quant_type, inplace=False)
        print("model", model)
        print("sq_model", sq_model)

        mesure = sq.mesure(model)
        sq_mesure = sq.mesure(sq_model)
        
        print("Total bits of information in model: ", mesure)
        print("Total bits of information in super quantized model: ", sq_mesure)

        print("Active total bits of information in model: ", mesure - 32 * 128 * depth * (2*vocab_size - seq_length))
        print("Active total bits of information in super quantized model: ", sq_mesure - 32 * 128 * depth * (2*vocab_size - seq_length))

        train(model)
        train(sq_model)

def train(model):
    train_dataloader = DataLoader(train_dataset, batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size)
    recipe = ProgressiveRecipes(model)
    recipe.base_recipe(epochs=40, iterations=10, global_trainning=0, scaling_factor=1, constructive=True)
    trainer = ProgressiveTrainer(recipe)
    trainer.train(
        train_dataloader, optim.AdamW, nn.CrossEntropyLoss(), validation_dataloader
    )

if __name__ == "__main__":
    compare_models()