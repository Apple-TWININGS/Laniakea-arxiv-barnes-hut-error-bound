"""Pytest configuration: register markers and set a CPU-safe default device."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running smoke / scale tests")


@pytest.fixture
def device():
    import torch

    return torch.device("cpu")
