"""Disjoint synthetic sequence tasks and validated local parallel-text adapter."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from research_lab.config import ExperimentConfig, Pair

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class Vocabulary:
    tokens: list[str]

    @classmethod
    def fit(cls, pairs: list[Pair]) -> "Vocabulary":
        words = {word for pair in pairs for text in (pair.source, pair.target) for word in text.split()}
        if words.intersection(SPECIAL):
            raise ValueError("reserved special tokens are not allowed in data")
        if len(words) > 4092:
            raise ValueError("local vocabulary exceeds 4096-token budget")
        return cls(SPECIAL + sorted(words))

    def encode(self, text: str, boundaries: bool = True) -> list[int]:
        if not text.strip():
            raise ValueError("input must contain at least one token")
        lookup = {word: index for index, word in enumerate(self.tokens)}
        ids = [lookup.get(word, UNK) for word in text.split()]
        return [BOS, *ids, EOS] if boundaries else ids

    def decode(self, ids: list[int]) -> list[str]:
        result = []
        for index in ids:
            if index == EOS:
                break
            if index not in (PAD, BOS):
                result.append(self.tokens[index])
        return result


@dataclass
class SequenceDataset:
    train: list[Pair]
    validation: list[Pair]
    test: list[Pair]
    vocab: Vocabulary
    task: str

    def statistics(self) -> dict[str, Any]:
        serialized = self.serialize()
        return {
            "version": "sequence-adapter-v1", "task": self.task,
            "split_sizes": {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)},
            "vocabulary_size": len(self.vocab.tokens),
            "sha256": hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(),
            "preprocessing": "whitespace; joint vocabulary fitted on training only",
            "license": "project-generated" if self.task != "parallel" else "user-supplied, not inferred",
        }

    def serialize(self) -> dict[str, Any]:
        return {name: [pair.model_dump() for pair in getattr(self, name)] for name in ("train", "validation", "test")}

    def batch(self, pairs: list[Pair], device: str = "cpu") -> tuple[Tensor, Tensor]:
        return pad([self.vocab.encode(pair.source) for pair in pairs], device), pad(
            [self.vocab.encode(pair.target) for pair in pairs], device
        )

    def majority_baseline(self) -> str:
        return Counter(word for pair in self.train for word in pair.target.split()).most_common(1)[0][0]


def pad(sequences: list[list[int]], device: str = "cpu") -> Tensor:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("cannot batch empty sequences")
    result = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        result[i, :len(sequence)] = torch.tensor(sequence, device=device)
    return result


def build_dataset(config: ExperimentConfig) -> SequenceDataset:
    rng = random.Random(config.seed)
    if config.task == "parallel":
        pairs = config.pairs or []
        sources = [pair.source for pair in pairs]
        if len(set(sources)) != len(sources):
            raise ValueError("duplicate source sequences risk split leakage; remove duplicates")
        if any(max(len(pair.source.split()), len(pair.target.split())) > config.max_length for pair in pairs):
            raise ValueError("uploaded sequence exceeds configured max_length")
        pairs = list(pairs)
        rng.shuffle(pairs)
        train_end, val_end = int(len(pairs) * 0.8), int(len(pairs) * 0.9)
        train, validation, test = pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]
    else:
        total = config.train_size + 2 * config.eval_size
        space = sum(config.symbols**length for length in range(config.min_length, config.max_length + 1))
        if total > space:
            raise ValueError("requested split sizes exceed number of unique possible sequences")
        seen: set[tuple[int, ...]] = set()
        pairs = []
        attempts = 0
        while len(pairs) < total:
            attempts += 1
            if attempts > total * 1000:
                raise ValueError("unique sampling exhausted; increase symbols/length")
            seq = tuple(rng.randrange(config.symbols) for _ in range(rng.randint(config.min_length, config.max_length)))
            if seq in seen:
                continue
            seen.add(seq)
            target = seq if config.task == "copy" else seq[::-1]
            pairs.append(Pair(source=" ".join(map(str, seq)), target=" ".join(map(str, target))))
        train = pairs[:config.train_size]
        validation = pairs[config.train_size:config.train_size + config.eval_size]
        test = pairs[config.train_size + config.eval_size:]
    return SequenceDataset(train, validation, test, Vocabulary.fit(train), config.task)
