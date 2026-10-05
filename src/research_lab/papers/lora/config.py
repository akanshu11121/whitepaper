"""Strict, bounded LoRA experiment configurations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoraExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(42, ge=0, le=2**32 - 1)
    input_dim: int = Field(32, ge=2, le=256)
    output_dim: int = Field(16, ge=2, le=256)
    rank: int = Field(2, ge=1, le=128)
    target_rank: int = Field(2, ge=1, le=128)
    alpha: float | None = Field(None, gt=0, le=1024)
    target_alpha: float | None = Field(None, gt=0, le=1024)
    lora_dropout: float = Field(0, ge=0, lt=1)
    init_std: float = Field(0.02, gt=0, le=1)
    bias_mode: Literal["none", "all"] = "none"
    steps: int = Field(300, ge=1, le=5000)
    batch_size: int = Field(32, ge=1, le=256)
    train_size: int = Field(512, ge=16, le=4096)
    eval_size: int = Field(128, ge=8, le=1024)
    learning_rate: float = Field(0.05, gt=0, le=10)
    target_scale: float = Field(1, gt=0, le=10)
    noise_std: float = Field(0, ge=0, le=1)
    device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_rank(self) -> "LoraExperimentConfig":
        if self.rank > min(self.input_dim, self.output_dim):
            raise ValueError("rank must not exceed min(input_dim, output_dim)")
        if self.target_rank > min(self.input_dim, self.output_dim):
            raise ValueError("target_rank must not exceed min(input_dim, output_dim)")
        return self

    @property
    def effective_alpha(self) -> float:
        return self.alpha if self.alpha is not None else float(self.rank)

    @property
    def effective_target_alpha(self) -> float:
        return self.target_alpha if self.target_alpha is not None else float(self.target_rank)


class LoraBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(42, ge=0, le=2**32 - 1)
    input_dim: int = Field(1024, ge=2, le=4096)
    output_dim: int = Field(1024, ge=2, le=4096)
    rank: int = Field(8, ge=1, le=256)
    alpha: float | None = Field(None, gt=0, le=4096)
    batch_size: int = Field(1, ge=1, le=128)
    repeats: int = Field(30, ge=5, le=200)
    warmup: int = Field(5, ge=1, le=50)
    dtype: Literal["float32", "float64"] = "float32"

    @model_validator(mode="after")
    def validate_rank(self) -> "LoraBenchmarkConfig":
        if self.rank > min(self.input_dim, self.output_dim):
            raise ValueError("rank must not exceed min(input_dim, output_dim)")
        return self
