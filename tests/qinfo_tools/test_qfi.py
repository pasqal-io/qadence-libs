from __future__ import annotations

import numpy as np
import random
import torch
from torch import Size, allclose

from qadence import QuantumCircuit, BasisSet
from qadence.constructors import hea, feature_map
from qadence.operations import RX

from qadence_libs.qinfo_tools import get_quantum_fisher, get_quantum_fisher_spsa


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def create_hea_circuit(n_qubits, layers):
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX], periodic=True)
    circuit = QuantumCircuit(n_qubits, ansatz, fm)
    return circuit


# Textbook QFI matrix for create_hea_circuit(2, 2)
textbook_qfi = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)


def test_qfi_exact():
    circuit = create_hea_circuit(2, 2)
    vparams_vals = [torch.Tensor([1.0]) for vparam in circuit.parameters() if vparam.trainable]
    fm_dict = {"phi": torch.Tensor([0])}
    qfi_mat_exact = get_quantum_fisher(circuit, vparams_values=vparams_vals, fm_dict=fm_dict)

    n_vparams = len(vparams_vals)
    assert qfi_mat_exact.shape == Size([n_vparams, n_vparams])
    assert allclose(textbook_qfi, qfi_mat_exact)


def test_qfi_spsa():
    circuit = create_hea_circuit(2, 2)
    vparams_vals = [torch.Tensor([1.0]) for vparam in circuit.parameters() if vparam.trainable]
    fm_dict = {"phi": torch.Tensor([0])}

    qfi_mat_spsa = None
    for iteration in range(50):
        qfi_mat_spsa, qfi_positive_sd = get_quantum_fisher_spsa(
            circuit,
            iteration,
            vparams_values=vparams_vals,
            fm_dict=fm_dict,
            previous_qfi_estimator=qfi_mat_spsa,
            beta=0.1,
            epsilon=0.01,
        )
        if iteration == 1:
            initial_nrm = torch.linalg.norm(textbook_qfi - qfi_mat_spsa)

    final_nrm = torch.linalg.norm(textbook_qfi - qfi_mat_spsa)
    assert 2 * final_nrm < initial_nrm

    n_vparams = len(vparams_vals)
    assert qfi_mat_spsa.shape == Size([n_vparams, n_vparams])
    assert qfi_positive_sd.shape == Size([n_vparams, n_vparams])

    # Check that qfi_positive_sd is positive semi-definite
    eigvals = torch.linalg.eigh(qfi_positive_sd)[0]
    assert torch.all(eigvals >= 0)
