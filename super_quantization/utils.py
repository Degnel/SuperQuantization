from torch import nn
from super_quantization.super_quantization import QuantizedLayer

QUANTIZE_COEFS = {
    "01": 32,
    "_11": 32,
    "_2_112": 16,
    "_2_101": 16,
    "_1012": 16,
}

def mesure(model: nn.Module):
    total_bits = 0
    leaf_modules = _iterate_mod_tree(model)
    for module in leaf_modules:
        for name, p in module.named_parameters():
            if p.requires_grad:
                if module.__class__ is QuantizedLayer and name == "weight":
                    coef = QUANTIZE_COEFS[module.quantize_mode]
                else:
                    coef = 1
                element_size_in_bits = p.element_size() * 8 // coef
                total_bits += p.numel() * element_size_in_bits

    return int(total_bits)


def _iterate_mod_tree(module: nn.Module):
    children = set()
    if list(module.named_children()):
        for _, child in module.named_children():
            children |= _iterate_mod_tree(child)
    else:
        children |= {module}

    return children