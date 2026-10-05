import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_threads():
    torch.set_num_threads(1)
