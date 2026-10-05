"""Validated, bounded local experiment configuration (engineering decisions)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    d_model: int = Field(64, ge=8, le=1024)
    heads: int = Field(4, ge=1, le=32)
    layers: int = Field(2, ge=1, le=8)
    d_ff: int = Field(128, ge=8, le=4096)
    dropout: float = Field(0.1, ge=0, lt=1)
    attention_dropout: float = Field(0, ge=0, lt=1)
    positions: Literal["sinusoidal", "learned", "none"] = "sinusoidal"
    max_length: int = Field(128, ge=8, le=2048)
    optimized: bool = False

    @model_validator(mode="after")
    def divisible(self) -> "ModelConfig":
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self


def default_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=64, heads=4, layers=2, d_ff=128, dropout=0.1,
        attention_dropout=0, positions="sinusoidal", max_length=128, optimized=False,
    )


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1, max_length=4096)
    target: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def nonempty(self) -> "Pair":
        if not self.source.strip() or not self.target.strip():
            raise ValueError("source and target must contain tokens")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(42, ge=0, le=2**32 - 1)
    task: Literal["copy", "reverse", "parallel"] = "reverse"
    model_type: Literal["transformer", "gru"] = "transformer"
    model: ModelConfig = Field(default_factory=default_model_config)
    steps: int = Field(600, ge=1, le=5000)
    batch_size: int = Field(32, ge=1, le=64)
    train_size: int = Field(512, ge=16, le=4096)
    eval_size: int = Field(64, ge=8, le=256)
    min_length: int = Field(3, ge=1, le=32)
    max_length: int = Field(8, ge=1, le=48)
    symbols: int = Field(12, ge=2, le=128)
    warmup_steps: int = Field(100, ge=1, le=4000)
    learning_rate_factor: float = Field(0.5, gt=0, le=5)
    smoothing: float = Field(0.1, ge=0, lt=1)
    device: Literal["cpu", "cuda"] = "cpu"
    pairs: list[Pair] | None = Field(None, max_length=4096)

    @model_validator(mode="after")
    def budget(self) -> "ExperimentConfig":
        if self.min_length > self.max_length:
            raise ValueError("min_length must not exceed max_length")
        if self.max_length + 2 > self.model.max_length:
            raise ValueError("model.max_length must accommodate tokens plus BOS/EOS")
        if self.model.d_model > 256 or self.model.d_ff > 1024 or self.model.layers > 4:
            raise ValueError("local lab training is bounded to width 256, FFN 1024, depth 4")
        if self.task == "parallel" and (not self.pairs or len(self.pairs) < 32):
            raise ValueError("parallel task requires at least 32 unique source/target pairs")
        if self.task != "parallel" and self.pairs is not None:
            raise ValueError("pairs are only supported for parallel tasks")
        return self


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(42, ge=0, le=2**32 - 1)
    length: int = Field(32, ge=2, le=128)
    batch_size: int = Field(4, ge=1, le=16)
    d_model: int = Field(64, ge=8, le=256)
    heads: int = Field(4, ge=1, le=16)
    repeats: int = Field(30, ge=5, le=200)
    warmup: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def divisible(self) -> "BenchmarkConfig":
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self
