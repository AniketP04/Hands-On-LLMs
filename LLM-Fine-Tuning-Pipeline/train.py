"""
This module handles:
  1. Tokenizing / preprocessing the raw dataset (prompt + completion → token IDs).
  2. Configuring the Hugging Face TrainingArguments.
  3. Running the Trainer loop and saving the fine-tuned model.
 
The training uses a simple causal language-modeling objective: the model is
trained to predict each token given all previous tokens (including the prompt),
so both prompt and completion tokens contribute to the loss.
 
Usage:
    from train import train
    train(model, tokenizer, dataset)

"""
 
from config import Config
from transformers import TrainingArguments, Trainer

def preprocess_function(examples, tokenizer):

    """
    Tokenize a batch of (prompt, completion) pairs into model input format.
 
    The prompt and completion are concatenated with a newline separator so the
    model sees: "<prompt>\\n<completion>" as a single sequence. The sequence is
    then tokenized, padded, and truncated to Config.MAX_LENGTH.
 
    Because this is a causal LM objective, the 'labels' are set equal to
    'input_ids' so the model learns to predict every token — including those
    in the prompt. If you want the model to only learn the completion tokens,
    you would mask out the prompt positions in 'labels' by setting them to -100.
 
    Args:
        examples (dict):
            A batch dict with keys:
              'prompt'     (list[str]): Input text / instruction strings.
              'completion' (list[str]): Target output strings.
            Provided automatically by datasets.Dataset.map().
 
        tokenizer:
            A Hugging Face tokenizer whose __call__ method accepts text and
            returns a dict with 'input_ids', 'attention_mask', etc.
 
    Returns:
        dict:
            Tokenizer output dict with the following keys added:
              'input_ids'      (list[list[int]]): Token ID sequences.
              'attention_mask' (list[list[int]]): 1 for real tokens, 0 for padding.
              'labels'         (list[list[int]]): Copy of input_ids used as training targets.
    """

    inputs = [f"{prompt}\n{completion}" for prompt, completion in zip(examples["prompt"],
                                                                      examples["completion"])]
    tokenized = tokenizer(
        inputs, 
        padding= "max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized


def train(model, tokenizer, dataset):

    """
    Run the full training loop: preprocess dataset → configure trainer → train → save.
 
    This function:
      1. Tokenizes the entire dataset using preprocess_function (batched for speed).
      2. Sets up TrainingArguments using values from Config.
      3. Creates a Hugging Face Trainer and calls .train().
      4. Saves the fine-tuned model and tokenizer to Config.OUTPUT_DIR.
 
    Args:
        model:
            A Hugging Face (PEFT-wrapped or full) causal LM ready for training.
            Should already be on the correct device(s).
 
        tokenizer:
            The tokenizer matching the model. Used for padding/truncation
            during preprocessing and saved alongside the model after training.
 
        dataset (datasets.Dataset):
            The raw Hugging Face dataset with 'prompt' and 'completion' columns.
            Output of dataset.get_dataset().
 
    Returns:
        None.
            The trained model is saved to disk at Config.OUTPUT_DIR.
    """
    
    print(" ----- Preprocessing Dataset -----")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched =True
    )

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(Config.OUTPUT_DIR)