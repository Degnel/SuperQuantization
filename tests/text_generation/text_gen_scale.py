import sys
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim, nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.text_generation.transformer import Transformer
from super_quantization.super_quantizer import SuperQuantizer
from tests.text_generation.preprocessing import get_data
from frEase.recipes import ProgressiveRecipes
from frEase.trainer import ProgressiveTrainer

seq_length = 5
vocab_size = 10000
train_batch_count = 2000
test_batch_count = 2000
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sq = SuperQuantizer()
train_dataset, validation_dataset, vocab = get_data(seq_length, vocab_size, train_batch_count, test_batch_count)

depths = [2, 4, 8, 16]
quant_types = {
    "att": {"K": "01", "Q": "01"},
    "out": {"O": "01"},
    "val": {"V": "01"},
    "fc_1": {"fc_1": "_1012"},
}

def compare_models():
    test_losses = {}
    for depth in depths:
        model = Transformer(
            d_model=128 * depth,
            n_heads=depth,
            d_ff=512 * depth,
            depth=depth,
            vocab_size=vocab_size,
            max_context_size=seq_length,
        ).to(device)
        models = {"base": model}
        for key, qtype in quant_types.items():
            models[key] = sq.quantize(model, qtype, inplace=False)
        
        for name, mod in models.items():
            _, test_loss = train(mod, name)
            test_losses.setdefault(name, []).append((depth, test_loss))
    
    plot_test_loss(test_losses)

def train(model, name):
    dataloader_args = {"batch_size": batch_size}
    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    validation_dataloader = DataLoader(validation_dataset, **dataloader_args)
    recipe = ProgressiveRecipes(model)

    if name == "base":
        recipe.base_recipe(epochs=400, global_trainning=1, constructive=False)
    else:
        recipe.base_recipe(epochs=40, iterations=10, global_trainning=0, scaling_factor=1)
    
    trainer = ProgressiveTrainer(recipe)
    name = f"./results/{name}.pkl"
    return trainer.train(
        train_dataloader, optim.AdamW, nn.CrossEntropyLoss(), validation_dataloader, results_saving_name=name
    )

def plot_test_loss(test_losses):
    for name, values in test_losses.items():
        depths, losses = zip(*values)
        plt.plot(depths, losses, marker='o', label=name)
    plt.xlabel("Depth")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.title("Test Loss vs Depth")
    plt.show()

if __name__ == "__main__":
    compare_models()