import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.transformer import Transformer
from super_quantization.utils import mesure
from tests.text_generation.preprocessing import get_data
from torch.utils.data import DataLoader

seq_length = 5
vocab_size = 10000
train_dataset, validation_dataset, _ = get_data(seq_length, vocab_size)
train_dataloader = DataLoader(train_dataset, batch_size=16)
validation_dataloader = DataLoader(validation_dataset, batch_size=None)

# On entraine le transformer classique
transformer_model = Transformer(
    d_model=512,
    n_heads=8,
    d_ff=256,
    depth=12,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=True,
    quantize_fc_2=True,
    max_context_size=seq_length
)

print("Total bits of information in model: ", mesure(transformer_model))

transformer_model.train_model(train_dataloader)
transformer_loss = transformer_model.test_model(validation_dataloader)

print(transformer_loss)

# On entraine notre transformer
sq_model = Transformer(
    d_model=14,
    n_heads=3 * 16,
    d_ff=56,
    depth=12,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=False,
    quantize_fc_2=False,
    max_context_size=seq_length
)

print("Total bits of information in super quantized model: ", mesure(sq_model))
sq_model.train_model(train_dataloader)
sq_loss = sq_model.test_model(validation_dataloader)

print(sq_loss)