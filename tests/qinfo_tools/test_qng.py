from __future__ import annotations

import pytest
import torch
from torch import Size, allclose
import numpy as np
import random

from qadence import QNN, QuantumCircuit, BasisSet, hamiltonian_factory
from qadence.constructors import hea, feature_map
from qadence.operations import Z, RX, RY

from qadence_libs.qinfo_tools import QNG, QNG_SPSA

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def quadratic_dataset(samples):
    x_train = torch.rand(samples)
    return x_train, x_train**2


def sin_dataset(samples):
    x_train = torch.rand(samples)
    return x_train, torch.sin(x_train)


def create_hea_model(n_qubits, layers):
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX, RY], periodic=True)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs = hamiltonian_factory(n_qubits, detuning=Z)
    model = QNN(circuit, [obs])
    return circuit, model


# Create dummy quantum model for tests
TINY_CIRCUIT, TINY_QNN = create_hea_model(2, 2)

OPTIMIZERS_CONFIG = [
    (QNG, {"lr": 0.05, "beta": 10e-3}, 20),
    (QNG_SPSA, {"lr": 0.01, "beta": 10e-3, "epsilon": 0.01}, 20),
]
MODEL_CONFIG = [(TINY_CIRCUIT, TINY_QNN)]
DATASETS = [quadratic_dataset(100), sin_dataset(100)]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("optim_config", OPTIMIZERS_CONFIG)
@pytest.mark.parametrize("model_config", MODEL_CONFIG)
def test_optims(dataset, optim_config, model_config):

    circuit, model = model_config
    model.reset_vparams(torch.rand((len(model.vparams))))

    optim_class, config, iters = optim_config
    x_train, y_train = dataset
    mse_loss = torch.nn.MSELoss()
    vparams = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim_class(params=vparams, circuit=circuit, **config)
    initial_loss = mse_loss(model(x_train).squeeze(), y_train.squeeze())
    for i in range(iters):
        optimizer.zero_grad()
        loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

    assert initial_loss > 2 * loss
    if hasattr(optimizer, "current_iteration"):
        assert optimizer.current_iteration == iters
