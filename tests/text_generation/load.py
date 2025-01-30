import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.transformer import Transformer
from tests.text_generation.preprocessing import get_data
from torch.utils.data import DataLoader
import torch

seq_length = 5
vocab_size = 10000
_, validation_dataset, _ = get_data(seq_length, vocab_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer_model = Transformer(
    d_model=512,
    n_heads=8,
    d_ff=256,
    depth=12,
    quantize_Q=False,
    quantize_K=False,
    quantize_V=False,
    quantize_fc_1=False,
    quantize_fc_2=False,
    vocab_size=vocab_size,
    max_context_size=seq_length,
).to(device)

# state_dict = torch.load("./data/weights/model_weights.pth", map_location=torch.device('cpu'))
# transformer_model.load_state_dict(state_dict)
# transformer_model.test_model(validation_dataloader, mini_batch_count=100)

sq_model = Transformer(
    d_model=512,
    n_heads=8,
    d_ff=256,
    depth=12,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=True,
    quantize_fc_2=True,
    vocab_size=vocab_size,
    max_context_size=seq_length,
).to(device)

state_dict = torch.load(
    "./data/weights/sq_model_weights.pth", map_location=torch.device("cpu")
)
for key in state_dict:
    if "weight" in key:  # S'assurer que c'est un poids (exclut biais, etc.)
        if "embedding.weight" in key or "output_projection.weight" in key:
            continue
        param = state_dict[key]
        if param.ndim == 2:
            state_dict[key] = param.t()
sq_model.load_state_dict(state_dict)
sq_model.test_model(validation_dataloader, mini_batch_count=100)
