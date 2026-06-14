"""
This module provides three strategies for loading the base language model:
 
  1. load_model_with_sft()   — Standard supervised fine-tuning (full params).
  2. load_model_with_lora()  — Parameter-efficient LoRA fine-tuning.
  3. load_model_with_qlora() — Quantized LoRA (4-bit) fine-tuning for low VRAM.
 
All loaders read hyperparameters from the shared Config singleton so that
changes in config.py propagate here automatically.
 
Usage:
    from model import load_tokenizer, load_model_with_lora
    tokenizer = load_tokenizer()
    model     = load_model_with_lora()

"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from config import Config

# Tokenizer
def load_tokenizer():

    """
    Load the tokenizer for the model specified by Config.MODEL_NAME.
 
    The tokenizer converts raw text into token IDs (and back).
    If the tokenizer does not define a padding token (common for decoder-only
    models like GPT-style), we fall back to using the end-of-sequence token
    as the pad token. This is required for batched training.
 
    Returns:
        transformers.PreTrainedTokenizer:
            The tokenizer associated with Config.MODEL_NAME.
 
    """

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)


    # many decoder-only models (GPT, Qwen, LLaMA) don't set a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token        # Setting pad_token = eos_token is the standard workaround for batched training.

    return tokenizer


def load_model_with_sft():

    """
    Load the base causal language model for full-parameter supervised fine-tuning (SFT).
 
    In SFT all model parameters are trainable — no adapter layers are added.
    This requires the most GPU memory but can achieve the best performance
    when sufficient hardware and data are available.
 
    The model is loaded with `device_map='auto'` which distributes layers
    across available GPUs (and CPU if needed) automatically.
 
    Returns:
        transformers.PreTrainedModel:
            The base causal LM with all parameters unfrozen.
 
    """

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto"
    )

    model.print_trainable_parameter()

    return model

def load_model_with_lora(load_in_4bit= True,
                         load_in_8bit= False):
    
    """
    Load the base causal LM and attach LoRA adapter layers on top of it.
 
    LoRA (Low-Rank Adaptation) inserts small trainable matrices (adapters) into
    specific linear layers of the frozen base model. Only the adapter weights are
    updated during training — the base model parameters remain unchanged.
 
    This reduces trainable parameters by ~99% compared to full SFT while
    achieving comparable quality on many tasks.
 
    LoRA adapter task types:
        CAUSAL_LM    — Text generation (decoder-only models: GPT, Qwen, LLaMA)
        SEQ_2_SEQ_LM — Encoder-decoder models (T5, BART)
        TOKEN_CLS    — Token classification (NER, POS tagging)
        SEQ_CLS      — Sequence classification (text classification)
 
    Args:
        load_in_4bit (bool): Load base model in 4-bit (for QLoRA — prefer load_model_with_qlora).
        load_in_8bit (bool): Load base model in 8-bit (LLM.int8() quantization).
 
    Returns:
        peft.PeftModel:
            The base model wrapped with LoRA adapters applied to TARGET_MODULES.
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r= Config.LORA_R,           # rank of low-rank matrices, controls the number of trainable parameters
        lora_alpha = Config.LORA_ALPHA,     # scaling factor for LoRA updates, scale = lora_alpha / r
        target_modules = Config.TARGET_MODULES,     # names of the linear layers in the model to apply LoRA adapters to
        lora_dropout= Config.LORA_DROPOUT,          # applying dropouts to LoRA layers
        bias= "none",           # bias parameter are not trained
        task_type= "CAUSAL_LM"      # specifying the model task
    )

def load_model_with_qlora(load_in_4bit= True,
                          load_in_8bit= False,
                          double_quant= True,
                          compute_dtype= 'float16',
                          quant_dtype= 'nf4'):
    
    """
    Load the base causal LM in 4-bit quantization and attach LoRA adapters (QLoRA).
 
    QLoRA (Quantized LoRA) combines:
      - 4-bit NormalFloat (NF4) quantization of the frozen base model weights.
      - Double quantization to further compress the quantization constants.
      - BF16/FP16 compute dtype for efficient matrix multiplications.
      - LoRA adapters trained in full precision on top of the quantized base.
 
    This enables fine-tuning of very large models (13B, 70B+) on consumer GPUs
    with limited VRAM (e.g. a 24 GB RTX 4090 can fine-tune a 13B model).
 
    Args:
        load_in_4bit (bool):
            If True, load base model weights as 4-bit quantized integers.
            Dramatically reduces VRAM; should be True for QLoRA.
 
        load_in_8bit (bool):
            If True, load in 8-bit instead of 4-bit. Mutually exclusive with
            load_in_4bit — set only one to True.
 
        double_quant (bool):
            Enable double quantization (quantize the quantization constants too).
            Saves an additional ~0.4 bits per parameter on average.
            Recommended: True.
 
        compute_dtype (str):
            The floating-point dtype used for computation (matrix multiplications).
            '  float16' → FP16, fastest on most consumer GPUs.
            'bfloat16' → BF16, numerically stabler but requires Ampere+ GPU.
            Quantized weights are dequantized to this dtype on-the-fly.
 
        quant_dtype (str):
            The quantization data type used to store the 4-bit weights.
            'nf4'  → NormalFloat4, designed for normally-distributed weights (recommended).
            'fp4'  → standard 4-bit floating point.
 
    Returns:
        peft.PeftModel:
            The 4-bit quantized base model with LoRA adapters attached and
            only the adapter weights set as trainable.
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= load_in_4bit,
        load_in_8bit= load_in_8bit,
        bnb_4bit_use_double_quant= double_quant,        # enables nested quantization where the quantization constants 
                                                        # are themselves quantized, saving ~0.4 bits/param extra.
        bnb_4bit_compute_dtype= compute_dtype,
        bnb_4bit_quant_type= quant_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto",
        quantization_config= bnb_config
    )

    lora_config = LoraConfig(
        r= Config.LORA_R,
        lora_alpha= Config.LORA_ALPHA,
        target_modules= Config.TARGET_MODULES,
        lora_dropout= Config.LORA_DROPOUT,
        bias= "none",
        task_type= "CAUSAL_LM"
    )

    model= get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model