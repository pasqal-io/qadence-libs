from __future__ import annotations

import pytest
import torch
from torch import Size, allclose

from qadence import QuantumCircuit, BasisSet
from qadence.constructors import hea, feature_map
from qadence.operations import *

from qadence_libs.qinfo_tools import get_quantum_fisher, get_quantum_fisher_spsa


def create_hea_circuit(n_qubits, layers):
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX], periodic=True)
    circuit = QuantumCircuit(n_qubits, ansatz, fm)
    return circuit


# Create dummy quantum circuit for tests
TINY_CIRCUIT = create_hea_circuit(2, 2)
VPARAMS_VALS = [torch.Tensor([1]) for vparam in TINY_CIRCUIT.parameters() if vparam.trainable]
FM_DICT = {"phi": torch.Tensor([0])}

# Textbook QFI matrix of the dummy circuit
textbook_qfi = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)


@pytest.mark.parametrize("circuit", [TINY_CIRCUIT])
@pytest.mark.parametrize("vparams_vals", [VPARAMS_VALS])
@pytest.mark.parametrize("fm_dict", [FM_DICT])
def test_qfi_exact(circuit, vparams_vals, fm_dict):
    n_vparams = len(vparams_vals)
    qfi_mat_exact = get_quantum_fisher(circuit, vparams_values=vparams_vals, fm_dict=fm_dict)
    assert qfi_mat_exact.shape == Size([n_vparams, n_vparams])
    assert allclose(textbook_qfi, qfi_mat_exact)


@pytest.mark.parametrize("circuit", [TINY_CIRCUIT])
@pytest.mark.parametrize("vparams_vals", [VPARAMS_VALS])
@pytest.mark.parametrize("fm_dict", [FM_DICT])
def test_qfi_spsa(circuit, vparams_vals, fm_dict):
    n_vparams = len(vparams_vals)
    qfi_mat_spsa = None
    qfi_mat_spsa, qfi_positive_sd = get_quantum_fisher_spsa(
        circuit,
        0,
        vparams_values=vparams_vals,
        fm_dict=fm_dict,
        previous_qfi_estimator=qfi_mat_spsa,
        beta=0.1,
        epsilon=0.01,
    )
    assert qfi_mat_spsa.shape == Size([n_vparams, n_vparams])
    assert qfi_positive_sd.shape == Size([n_vparams, n_vparams])

    # Check that qfi_positive_sd is positive semi-definite (~1s)
    eigvals = torch.linalg.eigh(qfi_positive_sd)[0]
    assert torch.all(eigvals >= 0)
