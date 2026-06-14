"""
This module is responsible for:
  1. Reading the training dataset from a JSONL file on disk.
  2. Validating each record against the TrainExample schema.
  3. Returning a Hugging Face `datasets.Dataset` object ready for use
     in the training pipeline.

Expected JSONL format (one JSON object per line):
    {"prompt": "Some input text", "completion": "Desired model output"}

Usage:
    from dataset import get_dataset
    dataset = get_dataset()

"""

import json
from pathlib import Path

from datasets import Dataset
from pydantic import BaseModel, ValidationError

from config import Config


# Data Schema

class TrainExample(BaseModel):
    """
    Pydantic model that defines the expected schema for a single training record.

    Attributes:
        prompt     (str): The input text / instruction given to the model.
        completion (str): The expected / target output the model should learn to generate.

    Pydantic validates that both fields are present and are strings.
    Any record missing these fields or having wrong types will raise a ValidationError.
    """

    prompt: str
    completion: str


# Dataset Loading Function

def get_dataset() -> Dataset:
    """
    Load and validate the training dataset from the JSONL file specified
    in Config.DATASET_PATH.

    The function:
      - Resolves the dataset path relative to Config.PROJECT_ROOT if needed.
      - Reads each non-empty line of the JSONL file.
      - Validates each line against the TrainExample schema.
      - Raises a ValueError with the line number if any record is malformed.
      - Returns a Hugging Face Dataset built from all valid records.

    Returns:
        datasets.Dataset:
            A Hugging Face Dataset with columns 'prompt' and 'completion'.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist at the resolved path.
        ValueError:
            If any record in the JSONL file fails schema validation,
            includes the line number and validation details.
        json.JSONDecodeError:
            If a line contains malformed JSON (not a valid JSON object).

    Example:
        >>> dataset = get_dataset()
        >>> print(dataset[0])
        {'prompt': 'Tell me a joke.', 'completion': 'Why did the chicken...'}
    """

    data_list: list[dict] = []
    dataset_path: Path = Config.DATASET_PATH

    if not dataset_path.is_absolute():
        dataset_path = Config.PROJECT_ROOT / dataset_path

    print("----- Loading Dataset -----")
    with dataset_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):

            if not line.strip():
                continue

            record: dict = json.loads(line)

            try:
                example = TrainExample(**record)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid dataset record at line {lineno}: {exc}"
                ) from exc

            data_list.append(example.dict())

    print(f"  Loaded {len(data_list)} training examples.")
    
    return Dataset.from_list(data_list)