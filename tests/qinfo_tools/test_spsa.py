from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from qadence import BasisSet, Overlap, QuantumCircuit
from qadence.constructors import feature_map, hea
from qadence.operations import RX, RY
from torch import Size

from qadence_libs.qinfo_tools.spsa import _shifted_overlap, spsa_2gradient_step

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def create_hea_circuit(n_qubits: int, layers: int) -> QuantumCircuit:
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX, RY], periodic=True)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    return circuit


@pytest.mark.parametrize("shift", [0.0, 0.1])
@pytest.mark.parametrize("phi", [0.0, 0.1])
def test_shifted_overlap(shift: float, phi: float) -> None:
    circuit = create_hea_circuit(2, 2)
    fm_dict = {"phi": torch.Tensor([phi])}

    ovrlp_model = Overlap(circuit, circuit)
    vparams_dict = {k: v for (k, v) in ovrlp_model._params.items() if v.requires_grad}

    shift_tensor = shift * torch.ones(len(vparams_dict))
    ovrlp = _shifted_overlap(
        ovrlp_model,
        shift_tensor,
        fm_dict,
        vparams_dict,
    )
    if shift == 0.0:
        assert torch.isclose(ovrlp, torch.ones_like(ovrlp))
    if shift == 0.1:
        assert not torch.isclose(ovrlp, torch.ones_like(ovrlp))


@pytest.mark.parametrize("epsilon", [0.01, 0.001])
def test_spsa_2gradient(epsilon: float) -> None:
    circuit = create_hea_circuit(2, 2)
    fm_dict = {"phi": torch.Tensor([0.0])}
    ovrlp_model = Overlap(circuit, circuit)

    hess_spsa = spsa_2gradient_step(ovrlp_model, epsilon, fm_dict)

    assert hess_spsa.shape == Size([ovrlp_model.num_vparams, ovrlp_model.num_vparams])
    assert torch.all(torch.isreal(hess_spsa)), "Hessian matrix is not real"
