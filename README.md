# SuperQuantization

SuperQuantization is a research-oriented repository for exploring quantization techniques in deep learning, with a focus on transformers and large language models (LLMs). The repository provides tools, models, and experiments for quantizing neural network weights.

## Features

- **Quantized Transformer Architectures**: Implementations of transformers with support for quantized layers (e.g., SQ2, SQ1, etc.).
- **Flexible Quantization Recipes**: Easily switch between full-precision and quantized models, or mix quantization strategies per layer.
- **Benchmarking & Comparison**: Tools for comparing quantized models to full-precision baselines in terms of accuracy, information content, and speed.
- **Dataset Utilities**: Scripts for preprocessing and tokenizing datasets (e.g., Wikipedia and CodeParrot) for language modeling and text generation tasks.
- **Integration with LLMs**: Support for quantizing and evaluating large models, including Llama and DeepSeek architectures.
- **Progressive Training**: Recipes and trainers for progressive quantization and training strategies. Those were not used for the final tests as we aimed to test one variable at a time. However, you can achieve notable speedup when enabled.

## Directory Structure

- `super_quantization/` — Core quantization logic and quantizer classes.
- `tests/` — Experimental scripts, model definitions, and evaluation tools for transformers, text generation, and LLMs.
- `src/` — C++ code for matrix multiplication benchmarking.
- `frEase/` — Progressive training recipes and trainers (for advanced training strategies).

## Data

You can get the datasets on Hugging Face:
- CodeParrot: https://huggingface.co/datasets/codeparrot/codeparrot-clean
- Wikipedia: https://huggingface.co/datasets/BEE-spoke-data/wikipedia-20230901.en-deduped

## Usage

### Comparing Model Architectures

```bash
uv run tests/transformer/text_gen_scale.py.py
```

### Benchmarking Matrix Multiplication

```bash
g++ -O3 src/mm.cpp -o mm && ./mm
```

## Requirements

- Python 3.8+
- uv (`pipx install uv` to have uv available anywhere on your computer)

Install dependencies:

```bash
uv sync
```

## Paper

You can read the paper [here](SuperQuantization.pdf).