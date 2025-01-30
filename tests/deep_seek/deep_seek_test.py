import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from super_quantization.super_quantizer import SuperQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True
)

try:
    print("Loading model from local directory...")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("deepseek-ai/deepseek-coder-1.3b-base", "config.json")
    dir = os.path.dirname(path)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        dir, trust_remote_code=True
    )
    print("Model loaded successfully from local directory.")
except Exception as e:
    print(f"Exception occurred: {e}")
    print("Trying to load model from Hugging Face Hub...")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True
    )
    print("Model loaded successfully from Hugging Face Hub.")

print("Quantizing model...")
sq = SuperQuantizer()
sq.quantize(model, {"self_attn.q_proj": "_11"})

print("Generating answer...")
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
pass