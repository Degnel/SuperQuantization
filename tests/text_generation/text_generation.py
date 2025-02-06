import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.transformer import Transformer
from super_quantization.super_quantizer import SuperQuantizer
from tests.text_generation.preprocessing import get_data
from torch.utils.data import DataLoader
import torch

seq_length = 5
vocab_size = 10000
mini_batch_count = 1000
train_dataset, validation_dataset, vocab = get_data(seq_length, vocab_size)
train_dataloader = DataLoader(train_dataset, batch_size=16)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sq = SuperQuantizer()

# # On entraine le transformer classique
# transformer_model = Transformer(
#     d_model=32,
#     n_heads=1,
#     d_ff=32,
#     depth=1,
#     quantize_Q=False,
#     quantize_K=False,
#     quantize_V=False,
#     quantize_fc_1=False,
#     quantize_fc_2=False,
#     vocab_size=vocab_size,
#     max_context_size=seq_length
# ).to(device)

# print("Total bits of information in model: ", sq.mesure(transformer_model))

# transformer_model.train_model(train_dataloader, mini_batch_count=mini_batch_count, epochs=60, lr=0.003)
# transformer_loss = transformer_model.test_model(validation_dataloader, mini_batch_count=100)
# # torch.save(transformer_model.state_dict(), "./drive/MyDrive/SuperQuantization/data/weights/model_weights.pth")

# print(transformer_loss)

# On entraine notre transformer
sq_model = Transformer(
    d_model=32,
    n_heads=2,
    d_ff=32,
    depth=2,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=True,
    quantize_fc_2=True,
    vocab_size=vocab_size,
    max_context_size=seq_length,
).to(device)


print("Total bits of information in super quantized model: ", sq.mesure(sq_model))
sq_model.train_model(
    train_dataloader, mini_batch_count=mini_batch_count, epochs=200, lr=0.001
)
sq_loss = sq_model.test_model(validation_dataloader, mini_batch_count=100)

torch.save(
    sq_model.state_dict(),
    "./drive/MyDrive/SuperQuantization/data/weights/sq_model_weights.pth",
)
# torch.save(sq_model.state_dict(), "./data/weights/model_weights.pth")

# inv_vocab = {v: k for k, v in vocab.items()}
# for x, y in validation_dataset:
#     print("Sample:")
#     words = [inv_vocab.get(idx.item(), '_') for idx in x]
#     print(words)

#     print("Real:")
#     test_words = [inv_vocab.get(idx.item(), '_') for idx in y]
#     print(test_words)

#     print("Predicted:")
#     pred_words = [inv_vocab.get(idx.item(), '_') for idx in sq_model(x.unsqueeze(0))[0].argmax(dim=0)]
#     print(pred_words)

print(sq_loss)
