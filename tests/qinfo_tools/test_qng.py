from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from qadence import QuantumCircuit
from torch import Tensor

from qadence_libs.qinfo_tools import QuantumNaturalGradient
from qadence_libs.types import FisherApproximation

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
    (
        {
            "lr": 0.1,
            "approximation": FisherApproximation.EXACT,
            "beta": 0.01,
        },
        20,
    ),
    (
        {
            "lr": 0.01,
            "approximation": FisherApproximation.SPSA,
            "beta": 0.1,
            "epsilon": 0.01,
        },
        20,
    ),
]
samples = 100
DATASETS = [quadratic_dataset(samples), sin_dataset(samples)]


def test_parameter_ordering(basic_optim_model: QuantumCircuit) -> None:
    model = basic_optim_model
    model.reset_vparams(torch.rand((len(model.vparams))))
    vparams_torch = [p.data for p in model.parameters() if p.requires_grad]
    vparams_qadence = [v.data for v in model.vparams.values()]
    assert len(vparams_torch) == len(vparams_qadence)
    msg = (
        "The ordering of the output of the `vparams()` method in QuantumModel"
        + "and the `parameters()` method in Torch is not consistent"
        + "for variational parameters."
    )
    assert vparams_torch == vparams_qadence, msg


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("optim_config", OPTIMIZERS_CONFIG)
def test_optims(
    dataset: tuple[Tensor, Tensor], optim_config: dict, basic_optim_model: QuantumCircuit
) -> None:
    model = basic_optim_model
    model.reset_vparams(torch.ones((len(model.vparams))))

    config, iters = optim_config
    x_train, y_train = dataset
    mse_loss = torch.nn.MSELoss()
    optimizer = QuantumNaturalGradient(model=model, **config)
    initial_loss = mse_loss(model(x_train).squeeze(), y_train.squeeze())
    for _ in range(iters):
        optimizer.zero_grad()
        loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

    assert initial_loss > 2.0 * loss

    if config["approximation"] == FisherApproximation.SPSA:
        assert optimizer.state["iter"] == iters
        assert optimizer.state["qfi_estimator"] is not None
