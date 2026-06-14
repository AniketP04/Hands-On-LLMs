"""
This module provides a lightweight qualitative evaluation of the fine-tuned
model by running it on a fixed set of test prompts and printing the generated
responses to stdout.
 
This is a qualitative (human-readable) evaluation, not a quantitative one
(no BLEU/ROUGE scores are computed here). It is intended as a quick sanity
check to see whether the model produces coherent, on-topic outputs after
fine-tuning.
 
Usage:
    from eval import evaluate_model
    evaluate_model(model, tokenizer)

"""

import torch
from config import Config

def evaluate_model(model, tokenizer):

    """
    Run the model on a set of test prompts and print prompt/response pairs.
 
    For each prompt the function:
      1. Tokenizes the prompt and moves tensors to the model's device.
      2. Calls model.generate() under torch.no_grad() (no gradient tracking).
      3. Decodes the generated token IDs back to a human-readable string.
      4. Prints the prompt and the full decoded response.
 
    Args:
        model:
            The fine-tuned (or LoRA/QLoRA-adapted) causal LM.
            Must have a .device attribute (provided by Hugging Face models).
 
        tokenizer:
            The tokenizer matching the model.
            Used to encode the prompts and decode the generated output.
 
        prompts (list[str], optional):
            List of prompt strings to evaluate.
            Defaults to DEFAULT_PROMPTS if not provided.
 
    Returns:
        None.
            Results are printed to stdout.
 
    """

    prompts = [
        "Write a short poem about the sea.",
        "Give me a recipe for chocolate cake.",
        "Explain the theory of relativity in simple terms."
    ]

    print("\n --- Running Evaluation:")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_OUTPUT_TOKEN,
                do_sample=True,
                top_p=Config.TOP_P,
                temperature=Config.TEMPERATURE
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}\nResponse: {decoded}")