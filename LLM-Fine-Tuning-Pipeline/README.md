# 🤖 LLM Fine-Tuning Pipeline

A clean, modular pipeline for fine-tuning large language models (LLMs) using **SFT**, **LoRA**, and **QLoRA** strategies. Built on top of Hugging Face `transformers`, `peft`, `datasets`, and `trl`. The default base model is **Qwen/Qwen1.5-1.8B**, but any Hugging Face causal LM can be plugged in.

---

## 📁 Project Structure

```
fine_tuning/
│
├── config.py           # Centralized configuration (hyperparams, paths, LoRA settings)
├── dataset.py          # Dataset loading & Pydantic validation from JSONL
├── model.py            # Model & tokenizer loaders (SFT / LoRA / QLoRA)
├── train.py            # Preprocessing, TrainingArguments, and Trainer loop
├── eval.py             # Qualitative evaluation on test prompts
├── main.py             # CLI entry point — wires everything together
├── requirements.txt    # Python dependencies
│
└── dataset/
    └── train.jsonl     # Training data (one JSON object per line)
```

---

## ⚙️ How It Works

```
main.py (CLI args)
    │
    ├──▶ config.py         → Loads & validates all hyperparameters
    ├──▶ model.py          → Loads tokenizer + model (SFT / LoRA / QLoRA)
    ├──▶ dataset.py        → Reads & validates train.jsonl → HF Dataset
    ├──▶ train.py          → Tokenizes dataset → TrainingArguments → Trainer.train()
    └──▶ eval.py           → Runs model on test prompts → prints responses
```

### Fine-Tuning Strategies

| Strategy | Description | VRAM Required |
|----------|-------------|---------------|
| `sft`    | Standard full-parameter supervised fine-tuning. All model weights are trainable. | High |
| `lora`   | LoRA adapters injected into selected linear layers. Only adapter weights train (~1% of params). | Medium |
| `qlora`  | LoRA on a 4-bit NF4 quantized base model. Enables fine-tuning large models on consumer GPUs. | Low |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fine_tuning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` requires a CUDA-capable GPU. On CPU-only machines, QLoRA and 4-bit loading will not be available.

### 3. Prepare Your Dataset

Create a `dataset/` folder and add a `train.jsonl` file. Each line must be a valid JSON object with exactly two fields:

```jsonl
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
{"prompt": "Write a haiku about the moon.", "completion": "Silver light descends,\nSilent guardian above,\nTides bow to its call."}
{"prompt": "Explain recursion simply.", "completion": "Recursion is when a function calls itself to solve a smaller version of the same problem."}
```

> ⚠️ Each record **must** contain both `prompt` and `completion` keys (strings). Any malformed record will raise a `ValueError` with the offending line number.

### 4. Run Training

```bash
python main.py --fine_tuning lora
```

That's it! The model will be saved to `./qwen-outputs/` after training.

---

## 🖥️ CLI Reference

All arguments are optional — defaults are pulled from `config.py`.

```bash
python main.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fine_tuning` | `str` | `lora` | Fine-tuning strategy: `sft`, `lora`, or `qlora` |
| `--model_name` | `str` | `Qwen/Qwen1.5-1.8B` | Any Hugging Face causal LM model ID |
| `--epochs` | `int` | `10` | Number of training epochs |
| `--batch_size` | `int` | `2` | Per-device training batch size |
| `--learning_rate` | `float` | `2e-5` | Optimizer learning rate |
| `--max_length` | `int` | `512` | Max token sequence length (prompt + completion) |
| `--max_output_token` | `int` | `50` | Max new tokens to generate during evaluation |
| `--temperature` | `float` | `0.7` | Sampling temperature (higher = more random) |
| `--top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `--output_dir` | `str` | `./qwen-outputs` | Directory to save the fine-tuned model |

### Examples

**LoRA fine-tuning (default, recommended):**
```bash
python main.py --fine_tuning lora --epochs 5 --batch_size 4
```

**QLoRA on a smaller GPU:**
```bash
python main.py --fine_tuning qlora --model_name Qwen/Qwen1.5-1.8B --epochs 3
```

**Full SFT with a different model:**
```bash
python main.py --fine_tuning sft --model_name gpt2 --learning_rate 5e-5 --epochs 2
```

**Custom output directory:**
```bash
python main.py --fine_tuning lora --output_dir ./my_model_checkpoints
```

---

## 🔧 Configuration (`config.py`)

All settings live in `config.py` as a `pydantic.BaseSettings` class. They can be overridden in three ways (in order of priority):

1. **CLI arguments** passed through `main.py`
2. **Environment variables** (e.g., `export MODEL_NAME=gpt2`)
3. **Defaults** defined in `config.py`

### Full Config Reference

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen1.5-1.8B` | Hugging Face model hub ID |
| `LORA_R` | `8` | LoRA rank — controls the size of adapter matrices |
| `LORA_ALPHA` | `16` | LoRA scaling factor (`scale = alpha / r`) |
| `LORA_DROPOUT` | `0.05` | Dropout probability on LoRA adapter layers |
| `TARGET_MODULES` | `['q_proj', 'v_proj']` | Model layers where LoRA adapters are injected |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `2` | Per-device batch size |
| `LEARNING_RATE` | `2e-5` | Learning rate |
| `MAX_LENGTH` | `512` | Max tokenized sequence length (truncated if exceeded) |
| `MAX_OUTPUT_TOKEN` | `50` | Max new tokens generated during evaluation |
| `TOP_P` | `0.9` | Nucleus sampling `p` threshold |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `OUTPUT_DIR` | `./qwen-outputs` | Where the fine-tuned model is saved |
| `DATASET_PATH` | `dataset/train.jsonl` | Path to the training JSONL file |
| `LOGGING_STEPS` | `1` | Log loss every N steps |
| `FP16` | `True` | Use FP16 mixed-precision training |

---

## 📦 Module Details

### `config.py` — Centralized Configuration
- Uses `pydantic.BaseSettings` for type validation and env-variable overriding.
- `validate_assignment = True` means any runtime change (e.g. via CLI) is also validated.
- Import the singleton: `from config import Config`

### `dataset.py` — Dataset Loader
- Reads `train.jsonl` line by line.
- Validates each record with the `TrainExample` Pydantic model (`prompt` + `completion`).
- Raises a descriptive `ValueError` with the line number if any record is malformed.
- Returns a `datasets.Dataset` object ready for `Trainer`.

### `model.py` — Model & Tokenizer Loaders
Three loader functions, all reading from `Config`:

- **`load_tokenizer()`** — Loads the tokenizer; sets `pad_token = eos_token` if missing (standard for decoder-only models).
- **`load_model_with_sft()`** — Full-parameter model load with `device_map="auto"`.
- **`load_model_with_lora()`** — Loads base model and wraps it with `LoraConfig` (using `TARGET_MODULES`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`).
- **`load_model_with_qlora()`** — Loads base model with `BitsAndBytesConfig` (4-bit NF4, double quantization, FP16 compute) then attaches LoRA adapters.

### `train.py` — Training Loop
- `preprocess_function()` — Concatenates `prompt\ncompletion`, tokenizes with padding/truncation to `MAX_LENGTH`, sets `labels = input_ids` (full causal LM objective).
- `train()` — Maps preprocessing over the dataset, sets up `TrainingArguments`, runs `Trainer.train()`, and saves the model with `trainer.save_model()`.

### `eval.py` — Qualitative Evaluation
- Runs the fine-tuned model on 3 built-in test prompts:
  - *"Write a short poem about the sea."*
  - *"Give me a recipe for chocolate cake."*
  - *"Explain the theory of relativity in simple terms."*
- Uses `torch.no_grad()` + `model.generate()` with `top_p` / `temperature` sampling.
- Prints prompt/response pairs to stdout — intended as a quick sanity check, not a quantitative benchmark.

### `main.py` — Entry Point
- Parses CLI arguments with `argparse`.
- Dynamically updates `Config` fields before anything loads.
- Orchestrates the full pipeline: tokenizer → model → dataset → train → eval.

---

## 📋 Requirements

```
transformers
datasets
peft
trl
bitsandbytes
accelerate
pydantic
```

Install all at once:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations

| Strategy | Minimum GPU VRAM | Recommended |
|---|---|---|
| `qlora` | ~6 GB | RTX 3060 / T4 |
| `lora`  | ~10 GB | RTX 3080 / A10 |
| `sft`   | ~16 GB+ | A100 / H100 |

> Training on CPU is possible but extremely slow and not recommended for models larger than ~125M parameters.

---

## 🗺️ Roadmap / Possible Extensions

- [ ] Add BLEU / ROUGE quantitative evaluation metrics in `eval.py`
- [ ] Add DPO (Direct Preference Optimization) support via `trl`
- [ ] Support multi-GPU training with `accelerate launch`
- [ ] Add train/validation split and per-epoch validation loss
- [ ] Mask prompt tokens in `labels` (set to `-100`) to train only on completions
- [ ] Add W&B / MLflow logging integration
- [ ] Dockerfile for reproducible environments

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT — Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Qwen1.5 Model](https://huggingface.co/Qwen/Qwen1.5-1.8B) — Alibaba Cloud
