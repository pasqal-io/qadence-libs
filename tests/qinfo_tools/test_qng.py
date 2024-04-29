from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from qadence import QuantumCircuit
from torch import Tensor

from qadence_libs.qinfo_tools import QuantumNaturalGradient, QuantumNaturalGradientSPSA

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def quadratic_dataset(samples: int) -> tuple[Tensor, Tensor]:
    x_train = torch.rand(samples)
    return x_train, x_train**2


def sin_dataset(samples: int) -> tuple[Tensor, Tensor]:
    x_train = torch.rand(samples)
    return x_train, torch.sin(x_train)


# Optimizers config [optim, config, iters]
OPTIMIZERS_CONFIG = [
    (QuantumNaturalGradient, {"lr": 0.1, "beta": 10e-2}, 20),
    (QuantumNaturalGradientSPSA, {"lr": 0.1, "beta": 10e-2, "epsilon": 0.01}, 20),
]
samples = 100
DATASETS = [quadratic_dataset(samples), sin_dataset(samples)]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("optim_config", OPTIMIZERS_CONFIG)
def test_optims(
    dataset: tuple[Tensor, Tensor], optim_config: dict, basic_optim_model: QuantumCircuit
) -> None:
    circuit, model = basic_optim_model
    model.reset_vparams(torch.rand((len(model.vparams))))

    optim_class, config, iters = optim_config
    x_train, y_train = dataset
    mse_loss = torch.nn.MSELoss()
    vparams = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim_class(params=vparams, circuit=circuit, **config)
    initial_loss = mse_loss(model(x_train).squeeze(), y_train.squeeze())
    for _ in range(iters):
        optimizer.zero_grad()
        loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

    assert initial_loss > 2.0 * loss
    if hasattr(optimizer, "current_iteration"):
        assert optimizer.current_iteration == iters
