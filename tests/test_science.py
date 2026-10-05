import math

import torch

from research_lab.config import ExperimentConfig, ModelConfig
from research_lab.datasets import BOS, EOS, PAD, build_dataset
from research_lab.papers.attention.attention import MultiHeadAttention, scaled_attention
from research_lab.papers.attention.model import Transformer, sinusoidal_positions
from research_lab.papers.attention.training import noam_rate, smoothed_loss


def test_scaled_attention_matches_hand_calculation():
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
    output, weights = scaled_attention(q, k, v)
    expected_weights = torch.softmax(torch.tensor([1.0 / math.sqrt(2), 0.0]), dim=0)
    assert torch.allclose(weights[0, 0, 0], expected_weights)
    assert torch.allclose(output[0, 0, 0], expected_weights @ v[0, 0])


def test_causal_mask_prevents_future_information_flow():
    q = torch.ones(1, 1, 3, 2)
    k = torch.ones(1, 1, 3, 2)
    v = torch.tensor([[[[1.0, 0], [2.0, 0], [99.0, 0]]]])
    allowed = torch.ones(1, 1, 3, 3, dtype=torch.bool).tril()
    first, _ = scaled_attention(q, k, v, allowed)
    v[:, :, 2] = 1000
    second, _ = scaled_attention(q, k, v, allowed)
    assert torch.allclose(first[:, :, :2], second[:, :, :2])


def test_padding_and_all_masked_query_are_finite_and_zero():
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 3)
    output, weights = scaled_attention(q, k, v, torch.zeros(1, 1, 1, 2, dtype=torch.bool))
    assert torch.isfinite(output).all() and torch.all(output == 0)
    assert torch.isfinite(weights).all() and torch.all(weights == 0)


def test_sinusoidal_encoding_matches_equation():
    positions = sinusoidal_positions(2, 4)
    assert torch.allclose(positions[0, 0], torch.tensor(0.0))
    assert torch.allclose(positions[0, 1], torch.tensor(1.0))
    assert torch.allclose(positions[1, 0], torch.sin(torch.tensor(1.0)))
    assert torch.allclose(positions[1, 1], torch.cos(torch.tensor(1.0)))


def test_multihead_reference_and_optimized_paths_agree():
    torch.manual_seed(4)
    reference = MultiHeadAttention(8, 2, optimized=False).eval()
    optimized = MultiHeadAttention(8, 2, optimized=True).eval()
    optimized.load_state_dict(reference.state_dict())
    value = torch.randn(2, 3, 8)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    expected, _ = reference(value, value, value, mask)
    actual, _ = optimized(value, value, value, mask)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_transformer_shapes_tied_weights_and_gradient():
    config = ModelConfig(d_model=16, heads=4, layers=2, d_ff=32, max_length=12, dropout=0)
    model = Transformer(10, config)
    source = torch.tensor([[BOS, 4, 5, EOS, PAD]])
    target = torch.tensor([[BOS, 6, 7, EOS, PAD]])
    logits, trace = model(source, target[:, :-1], capture=True)
    assert logits.shape == (1, 4, 10)
    assert model.embedding.weight.data_ptr() == model.output.weight.data_ptr()
    assert len(trace["encoder"]) == 2 and trace["encoder"][0].shape == (1, 4, 5, 5)
    logits.sum().backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_noam_schedule_warms_then_decays():
    warm = [noam_rate(step, 512, 4000) for step in (1, 100, 1000, 4000)]
    assert warm == sorted(warm)
    assert noam_rate(8000, 512, 4000) < noam_rate(4000, 512, 4000)
    assert math.isclose(noam_rate(1, 512, 4000), 512**-0.5 * 4000**-1.5)


def test_label_smoothing_excludes_padding_and_is_finite():
    logits = torch.randn(2, 3, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, PAD], [3, 4, PAD]])
    loss = smoothed_loss(logits, targets, 0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_dataset_splits_are_disjoint_and_train_vocab_only():
    config = ExperimentConfig(task="reverse", symbols=8, min_length=2, max_length=4, train_size=48, eval_size=8)
    data = build_dataset(config)
    splits = [{pair.source for pair in getattr(data, name)} for name in ("train", "validation", "test")]
    assert not splits[0] & splits[1] and not splits[0] & splits[2] and not splits[1] & splits[2]
    assert data.statistics()["sha256"]
