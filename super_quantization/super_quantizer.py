import copy
import torch
from torch import nn
from super_quantization.super_quantization import QuantizedLayer


class SuperQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.QUANTIZE_COEFS = {
            "01": 32,
            "_11": 32,
            "_2_112": 16,
            "_2_101": 16,
            "_1012": 16,
        }

    def quantize(
        self,
        model: nn.Module,
        layers_quant_type: dict = {},
        name: str = "model",
        inplace: bool = True,
    ):
        if not inplace:
            model = copy.deepcopy(model)
        leaf_modules = self._iterate_mod_tree(model, name)
        for name, module in leaf_modules:
            if module.__class__ is nn.Linear:
                for layer, quantize_mode in layers_quant_type.items():
                    if name.endswith(layer):
                        quantized_layer = QuantizedLayer(
                            input_dim=module.in_features,
                            output_dim=module.out_features,
                            bias=module.bias is not None,
                            quantize_mode=quantize_mode,
                        )
                        with torch.no_grad():
                            quantized_layer.weight.copy_(module.weight)
                            if module.bias is not None:
                                quantized_layer.bias.copy_(module.bias)
                        setattr(model, name, quantized_layer)
                        print('Quantizing layer: ', name)
        if not inplace:
            return model

    def mesure(self, model: nn.Module, name: str):
        total_bits = 0
        leaf_modules = self._iterate_mod_tree(model, name)
        for _, module in leaf_modules:
            for name, p in module.named_parameters():
                if p.requires_grad:
                    if module.__class__ is QuantizedLayer and name == "weight":
                        coef = self.QUANTIZE_COEFS[module.quantize_mode]
                    else:
                        coef = 1
                    element_size_in_bits = p.element_size() * 8 // coef
                    total_bits += p.numel() * element_size_in_bits

        return int(total_bits)

    def _iterate_mod_tree(self, module: nn.Module, name: str):
        children = set()
        if list(module.named_children()):
            for child_name, child in module.named_children():
                children |= self._iterate_mod_tree(child, name + "." + child_name)
        else:
            children |= {(name, module)}

        return children
