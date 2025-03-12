import sys
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim, nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from super_quantization.super_quantizer import SuperQuantizer
from tests.text_generation.transformer import Transformer
from tests.text_generation.preprocessing import get_data
from frEase.recipes import ProgressiveRecipes
from frEase.trainer import ProgressiveTrainer
import collections

wikipedia = False
seq_length = 256
vocab_size = 10000
train_batch_count = 500000
test_batch_count = 50000
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sq = SuperQuantizer()
train_dataset, validation_dataset, vocab = get_data(
    seq_length, vocab_size, train_batch_count, test_batch_count, wiki=wikipedia
)

# depths = [2, 4, 8, 16]
# depths = [1, 2, 3, 4]
depths = [2]
quant_types = {
    "fc": {"fc_1": "01", "fc_2": "01"},
    "fcv": {"V": "_1012", "fc_1": "01", "fc_2": "01"},
}


def compare_models():
    test_losses = {}
    for depth in depths:
        model = Transformer(
            # d_model=128*depth,
            d_model=32*depth,
            # n_heads=depth,
            n_heads=2,
            # d_ff=512*depth,
            d_ff=128*depth,
            depth=depth,
            vocab_size=vocab_size,
            max_context_size=seq_length,
            lora_ratio=4,
            rope=False,
            fc_quant_normalisation=True
        ).to(device)
        analyze_model_parameters(model)
        print("Params for base:", sum(p.numel() for p in model.parameters() if p.requires_grad) + model.d_model*(seq_length - vocab_size - 1))
        # models = {"base": model}
        models = {}

        for key, qtype in quant_types.items():
            models[key] = sq.quantize(model, qtype, inplace=False)
            mesure = sq.mesure(models[key])
            print(f"Mesure for model {key}: {int(mesure/32 + model.d_model*(seq_length - vocab_size - 1))}")

        for name, mod in models.items():
            _, test_loss = train(mod, name, depth)
            test_losses.setdefault(name, []).append((depth, test_loss))

    plot_test_loss(test_losses)


def analyze_model_parameters(model):
    param_groups = collections.defaultdict(float)
    total_params = 0
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        
        if "self_attention.Q" in name:
            key = "Q_weights" if "weight" in name else "Q_bias"
        elif "self_attention.K" in name:
            key = "K_weights" if "weight" in name else "K_bias"
        elif "self_attention.V" in name:
            key = "V_weights" if "weight" in name else "V_bias"
        elif "self_attention.O" in name:
            key = "O_weights" if "weight" in name else "O_bias"
        elif "fc_1" in name:
            key = "fc_1_weights" if "weight" in name else "fc_1_bias"
        elif "fc_2" in name:
            key = "fc_2_weights" if "weight" in name else "fc_2_bias"
        elif "layer_norm" in name:
            key = "LayerNorm_weights" if "weight" in name else "LayerNorm_bias"
        elif "embedding.weight" in name:
            key = "embedding.weight"
        else:
            key = name
        
        param_groups[key] += param_size

    param_distribution = {k: (v / total_params) * 100 for k, v in param_groups.items()}
    active_param_distribution = {k: (v / (total_params - param_groups["embedding.weight"])) * 100 for k, v in param_groups.items()}
    
    for param_type, percentage in param_distribution.items():
        print(f"{param_type}: {percentage:.2f}%")
    
    for param_type, percentage in active_param_distribution.items():
        print(f"Active {param_type}: {percentage:.2f}%")
    
    return param_distribution, active_param_distribution

def train(model, name, depth):
    dataloader_args = {"batch_size": batch_size}
    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    validation_dataloader = DataLoader(validation_dataset, **dataloader_args)
    recipe = ProgressiveRecipes(model)

    recipe.base_recipe(epochs=10, global_trainning=1, constructive=False)
    # if name == "base":
    #     recipe.base_recipe(epochs=400, global_trainning=1, constructive=False)
    #     #  recipe.base_recipe(epochs=1, global_trainning=1, constructive=False)
    # else:
    #     recipe.base_recipe(epochs=40, iterations=10, global_trainning=0, scaling_factor=1)
    #     # recipe.base_recipe(epochs=1, iterations=1, global_trainning=0, scaling_factor=1)
    trainer = ProgressiveTrainer(recipe)
    checkpoint_name = f"./checkpoints/{name}/depth_{depth}"
    res_name = f"./results/{name}/depth_{depth}.pkl"
    return trainer.train(
        train_dataloader,
        optim.AdamW,
        nn.CrossEntropyLoss(),
        validation_dataloader,
        show_batch_score=False,
        checkpoints_saving_path=checkpoint_name,
        results_saving_name=res_name,
    )


def plot_test_loss(test_losses):
    for name, values in test_losses.items():
        depths, losses = zip(*values)
        plt.plot(depths, losses, marker="o", label=name)
    plt.xlabel("Depth")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.title("Test Loss vs Depth")
    plt.show()


if __name__ == "__main__":
    compare_models()
