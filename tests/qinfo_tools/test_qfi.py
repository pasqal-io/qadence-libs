from __future__ import annotations

import random

import numpy as np
import torch
from qadence import QNN
from torch import Size, allclose

from qadence_libs.qinfo_tools import get_quantum_fisher, get_quantum_fisher_spsa
from qadence_libs.qinfo_tools.qfi import _positive_semidefinite_sqrt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Textbook QFI matrix for textbook_qfi_model with 2 qubits and 2 layers
textbook_qfi = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)


def test_positive_semidefinite_sqrt() -> None:
    # Construct a positive semidefinite matrix
    A = textbook_qfi @ textbook_qfi
    matrix_sqrt = _positive_semidefinite_sqrt(A)
    assert torch.allclose(matrix_sqrt @ matrix_sqrt, A)


def test_qfi_exact_no_params(textbook_qfi_model: QNN) -> None:
    # As long as FM parameters and variational parameters are independent,
    # the QFI matrix should be independent of the value of the FM params.
    circuit = textbook_qfi_model._circuit.abstract
    qfi_mat_exact = get_quantum_fisher(circuit)
    n_vparams = textbook_qfi_model.num_vparams
    assert qfi_mat_exact.shape == Size([n_vparams, n_vparams])
    assert allclose(textbook_qfi, qfi_mat_exact)


def test_qfi_exact_with_params(textbook_qfi_model: QNN) -> None:
    circuit = textbook_qfi_model._circuit.abstract
    vparams_dict = textbook_qfi_model.vparams
    fm_dict = {"phi": torch.Tensor([0])}
    qfi_mat_exact = get_quantum_fisher(circuit, vparams_dict=vparams_dict, fm_dict=fm_dict)
    n_vparams = len(vparams_dict)
    assert qfi_mat_exact.shape == Size([n_vparams, n_vparams])
    assert allclose(textbook_qfi, qfi_mat_exact)


def test_qfi_spsa(textbook_qfi_model: QNN) -> None:
    circuit = textbook_qfi_model._circuit.abstract
    vparams_dict = textbook_qfi_model.vparams
    fm_dict = {"phi": torch.Tensor([0])}
    qfi_mat_spsa = None
    for iteration in range(40):
        qfi_mat_spsa, qfi_positive_sd = get_quantum_fisher_spsa(
            circuit,
            iteration,
            vparams_dict=vparams_dict,
            fm_dict=fm_dict,
            previous_qfi_estimator=qfi_mat_spsa,
            beta=0.01,
            epsilon=0.01,
        )
        if iteration == 0:
            initial_nrm = torch.linalg.norm(textbook_qfi - qfi_mat_spsa)

    final_nrm = torch.linalg.norm(textbook_qfi - qfi_mat_spsa)
    assert final_nrm < initial_nrm

    n_vparams = len(vparams_dict)
    assert qfi_mat_spsa.shape == Size([n_vparams, n_vparams])  # type: ignore
    assert qfi_positive_sd.shape == Size([n_vparams, n_vparams])

    # Check that qfi_positive_sd is positive semi-definite
    eigvals = torch.linalg.eigh(qfi_positive_sd)[0]
    assert torch.all(eigvals >= 0)
