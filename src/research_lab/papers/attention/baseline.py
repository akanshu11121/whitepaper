"""Engineering baseline: vanilla GRU bottleneck seq2seq, not GNMT or paper's RNNs."""

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from research_lab.config import ModelConfig
from research_lab.datasets import PAD


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=PAD)
        self.encoder = nn.GRU(config.d_model, config.d_model, config.layers, batch_first=True)
        self.decoder = nn.GRU(config.d_model, config.d_model, config.layers, batch_first=True)
        self.output = nn.Linear(config.d_model, vocab_size)

    def encode(self, source: Tensor, capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        lengths = source.ne(PAD).sum(1).cpu()
        packed = pack_padded_sequence(self.embedding(source), lengths, batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        return state, {}

    def decode(self, target: Tensor, memory: Tensor, source: Tensor,
               capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        x, _ = self.decoder(self.embedding(target), memory)
        return self.output(x), {}

    def forward(self, source: Tensor, target: Tensor,
                capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        memory, _ = self.encode(source)
        return self.decode(target, memory, source)
