import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.llama_model import Transformer, ModelArgs
import collections


params = ModelArgs(
    dim=16384,
    n_layers=126,
    n_heads=128,
    n_kv_heads=8,
    vocab_size=128000,
    ffn_dim_multiplier=4
)

model = Transformer(params)

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
    for e in param_groups:
        print(e)
    param_distribution = {k: (v / total_params) * 100 for k, v in param_groups.items()}
    active_param_distribution = {k: (v / (total_params - param_groups["embedding.weight"])) * 100 for k, v in param_groups.items()}
    
    for param_type, percentage in param_distribution.items():
        print(f"{param_type}: {percentage:.2f}%")
    
    print()
    
    for param_type, percentage in active_param_distribution.items():
        if param_type != "embedding.weight":
            print(f"Active {param_type}: {percentage:.2f}%")
    
    return param_distribution, active_param_distribution


analyze_model_parameters(model)