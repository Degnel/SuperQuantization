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
        layers_quant_type: dict | str = "01",
        name: str | None = None,
        inplace: bool = True,
    ):
        if not inplace:
            model = copy.deepcopy(model)

        for module_name, module in model.named_children():
            full_name = f"{name}.{module_name}" if name else module_name
            quantize_type = self._quantize_type(full_name, layers_quant_type)
            if isinstance(module, nn.Linear) and quantize_type:
                quantized_layer = QuantizedLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    quantize_mode=quantize_type,
                )

                with torch.no_grad():
                    # quantized_layer.weight = nn.Parameter(
                    #     torch.zeros_like(quantized_layer.weight)
                    # )
                    quantized_layer.weight.copy_(module.weight.T)
                    if module.bias is not None:
                        quantized_layer.bias.copy_(module.bias)
                setattr(model, module_name, quantized_layer)
            elif isinstance(module, nn.Module):
                self.quantize(module, layers_quant_type, full_name, inplace=True)
        
        if not inplace:
            return model
    
    def _quantize_type(self, name, layers_quant_type):
        if isinstance(layers_quant_type, str):
            return layers_quant_type
        for layer, quantize_mode in layers_quant_type.items():
            if name.endswith(layer):
                return quantize_mode
        
        # leaf_modules = self._iterate_mod_tree(model)
        # for name, module in leaf_modules:
        #     if module.__class__ is nn.Linear:
        #         for layer, quantize_mode in layers_quant_type.items():
        #             if layers_quant_type == "all" or name.endswith(layer):
        #                 quantized_layer = QuantizedLayer(
        #                     in_features=module.in_features,
        #                     out_features=module.out_features,
        #                     bias=module.bias is not None,
        #                     quantize_mode=quantize_mode,
        #                 )
        #                 with torch.no_grad():
        #                     quantized_layer.weight = nn.Parameter(
        #                         # torch.zeros_like(quantized_layer.weight)
        #                         quantized_layer.weight.copy_(module.weight)
        #                     )
        #                     if module.bias is not None:
        #                         quantized_layer.bias.copy_(module.bias)
        #                 # setattr(model, name, quantized_layer)
        #                 parent_module = model
        #                 parent_name = name.split('.')
        #                 for part in parent_name[:-1]:
        #                     parent_module = getattr(parent_module, part)
        #                 setattr(parent_module, parent_name[-1], quantized_layer)
        #                 print("Quantizing layer: ", name)
        # if not inplace:
        #     return model

    def mesure(self, model: nn.Module):
        total_bits = 0
        leaf_modules = self._iterate_mod_tree(model)
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

    def _iterate_mod_tree(self, module: nn.Module, name: str | None = None):
        children = set()
        if list(module.named_children()):
            for child_name, child in module.named_children():
                children |= self._iterate_mod_tree(
                    child, f"{name}.{child_name}" if name else child_name
                )
        else:
            children |= {(name, module)}

        return children
