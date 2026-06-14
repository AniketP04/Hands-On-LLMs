"""
=============================================================================
config.py — Centralized Configuration Module
=============================================================================
 
This module defines the entire configuration for the fine-tuning pipeline
using Pydantic's BaseSettings. All hyperparameters, paths, and training
settings are stored here. Values can be overridden via environment variables
(because of BaseSettings) or programmatically at runtime via main.py CLI args.
 
Usage:
    from config import Config
    print(Config.MODEL_NAME)
=============================================================================
"""

from pathlib import Path

from pydantic import BaseSettings, Field


class BaseConfig(BaseSettings):

    """
    Central configuration class for the LLM fine-tuning pipeline.
 
    Inherits from pydantic's BaseSettings, which means every field can
    also be overridden by a matching environment variable (e.g. set
    MODEL_NAME=gpt2 in your shell and it will override the default below).
 
    Validation is applied automatically on assignment due to
    `validate_assignment = True` inside the inner Config class.
    """
    
    BASEDIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BASEDIR

    MODEL_NAME: str = 'Qwen/Qwen1.5-1.8B'

    LORA_R: int = Field(8, ge=1)
    LORA_ALPHA: int = Field(16, ge=1)
    LORA_DROPOUT: float = Field(0.05, ge=0.0, le=1.0)
    TARGET_MODULES: list[str] = Field(default_factory=lambda: ['q_proj', 'v_proj'])

    EPOCHS: int = Field(10, ge=1)
    BATCH_SIZE: int = Field(2, ge=1)
    LEARNING_RATE: float = Field(2e-5, gt=0)
    MAX_OUTPUT_TOKEN: int = Field(50, ge=1)
    TOP_P: float = Field(0.9, gt=0, le=1)
    TEMPERATURE: float = Field(0.7, gt=0)

    MAX_LENGTH: int = Field(512, ge=1)

    OUTPUT_DIR: Path = Path('./qwen-outputs')
    DATASET_PATH: Path = Path('dataset/train.jsonl')
    LOGGING_STEPS: int = Field(1, ge=1)
    FP16: bool = True

    class Config:
        env_prefix = ''
        validate_assignment = True


Config = BaseConfig()
