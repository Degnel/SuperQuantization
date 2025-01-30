import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from super_quantization.super_quantizer import SuperQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)

try:
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
except Exception:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("deepseek-ai/deepseek-coder-1.3b-base", "config.json")
    dir = os.path.dirname(path)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)

print('Quantizing model...')
sq = SuperQuantizer()
sq.quantize(model, {"self_attn.o_proj": "_11", "model.layers.23.self_attn.v_proj": "_2_112"})

print("Generating answer...")
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))