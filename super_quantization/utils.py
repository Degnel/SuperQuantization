import torch
from torch import nn
from super_quantization.super_quantization import QuantizedLayer

QUANTIZE_COEFS = {
    "01": 32,
    "_11": 32,
    "_2_112": 16,
    "_2_101": 16,
    "_1012": 16,
}


def mesure(model: nn.Module, name: str = "model"):
    def _iterate_mod_tree(module: nn.Module, name: str):
        children = set()
        if list(module.named_children()):
            for child_name, child in module.named_children():
                children |= _iterate_mod_tree(child, name + "." + child_name)
        else:
            children |= {(name, module)}

        return children

    total_bits = 0
    leaf_modules = _iterate_mod_tree(model, name)
    for _, module in leaf_modules:
        for name, p in module.named_parameters():
            if p.requires_grad:
                if module.__class__ is QuantizedLayer and name == "weight":
                    coef = QUANTIZE_COEFS[module.quantize_mode]
                else:
                    coef = 1
                element_size_in_bits = p.element_size() * 8 // coef
                total_bits += p.numel() * element_size_in_bits

    return int(total_bits)


def quantize_model(model: nn.Module, layers_quant_type: dict) -> nn.Module:
    """
    Remplace les couches linéaires par des QuantizedLayer en conservant les poids originaux.
    Seules les couches spécifiées dans layers_quant_type seront quantisées.
    """

    def should_quantize(layer_name: str) -> str | None:
        if layer_name in layers_quant_type:
            return layers_quant_type[layer_name]
        for key in layers_quant_type:
            if key in layer_name:
                return layers_quant_type[key]
        return None

    for name, module in model.named_children():
        quantize_mode = should_quantize(name)
        if isinstance(module, nn.Linear) and quantize_mode:
            # Créer une couche quantifiée avec les mêmes dimensions
            quantized_layer = QuantizedLayer(
                input_dim=module.in_features,
                output_dim=module.out_features,
                bias=module.bias is not None,
                quantize_mode=quantize_mode,
            )

            # Copier les poids et le biais
            with torch.no_grad():
                quantized_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    quantized_layer.bias.copy_(module.bias)

            # Remplacer la couche dans le modèle
            setattr(model, name, quantized_layer)

        # Appliquer récursivement aux sous-modules
        else:
            quantize_model(module, layers_quant_type)

    return model
