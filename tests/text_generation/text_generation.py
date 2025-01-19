import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.text_generation.transformer import Transformer
from super_quantization.utils import mesure
from tests.text_generation.preprocessing import get_data

X_train, y_train, X_test, y_test, _ = get_data()

# On entraine le transformer classique
transformer_model = Transformer(
    d_model=14,
    n_heads=3,
    d_ff=56,
    depth=12,
    quantize_Q=True,
    quantize_K=True,
    quantize_V=True,
    quantize_fc_1=True,
    quantize_fc_2=True,
)

print("Total bits of information in model: ", mesure(transformer_model))

transformer_model.train_model(X_train, y_train)
transformer_loss = transformer_model.train_model(X_test, y_test)

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
)

print("Total bits of information in super quantized model: ", mesure(sq_model))
sq_model.train_model(X_train, y_train)
sq_loss = sq_model.train_model(X_test, y_test)

print(sq_loss)

# Il faut rajouter un positionnal encoding dans la couche transformer
# Implémenter la méthode train directement dans la classe transformer
# Implémenter la méthode test directement dans la classe transformer
