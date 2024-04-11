from __future__ import annotations

from typing import Iterable

import torch
from qadence import Overlap
from qadence.blocks import parameters, primitive_blocks
from qadence.circuit import QuantumCircuit
from qadence.types import BackendName, DiffMode, OverlapMethod
from torch import Tensor

from qadence_libs.qinfo_tools.spsa import spsa_2gradient_step
from qadence_libs.qinfo_tools.utils import hessian


def _symsqrt(A: Tensor) -> Tensor:
    """Computes the square root of a Symmetric or Hermitian positive definite matrix.

    or batch of matrices.

    Code from https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def _set_circuit_vparams(circuit: QuantumCircuit, vparams_values: Iterable) -> None:
    """Sets the variational parameter values of the circuit."""
    if vparams_values is not None:
        blocks = primitive_blocks(circuit.block)
        iter_index = iter(range(len(blocks)))
        for block in blocks:
            params = parameters(block)
            for p in params:
                if p.trainable:
                    p.value = float(vparams_values[next(iter_index)])  # type: ignore


def _get_fm_dict(circuit: QuantumCircuit) -> dict:
    """Returns a dictionary holding the FM parameters of the circuit."""
    fm_dict = {}
    blocks = primitive_blocks(circuit.block)
    for block in blocks:
        params = parameters(block)
        for p in params:
            if not p.trainable:
                fm_dict[p.name] = torch.tensor([p.value])
    return fm_dict


def get_quantum_fisher(
    circuit: QuantumCircuit,
    vparams_values: Iterable[float | Tensor] | None = None,
    fm_dict: dict[str, Tensor] = dict(),
    backend: BackendName = BackendName.PYQTORCH,  # type: ignore
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,  # type: ignore
) -> Tensor:
    """Returns the exact Quantum Fisher Information (QFI) matrix of the quantum circuit.

    with given values for the variational parameters (vparams_values) and the
    feature map (fm_dict).

    Args:
        circuit (QuantumCircuit): The Quantum circuit we want to compute the QFI matrix of.
        vparams (tuple | list | Tensor | None):
            Values of the variational parameters where we want to compute the QFI.
        fm_dict (dict[str, Tensor]): Values of the feature map parameters.
        overlap_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.
    """

    # Get feature map dictionary (required to run Overlap().forward())
    if not fm_dict:
        fm_dict = _get_fm_dict(circuit)

    # Set the vparam_values
    if vparams_values is not None:
        _set_circuit_vparams(circuit, vparams_values)

    # Get Overlap() model
    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    # Run overlap model
    ovrlp = ovrlp_model(bra_param_values=fm_dict, ket_param_values=fm_dict)

    # Retrieve variational parameters of the overlap model
    # Importantly, the vparams of the overlap model are the vparams of the bra tensor,
    # Which means if we differentiate wrt vparams we are differentiating only wrt the
    # parameters in the bra and not in the ket
    vparams = [v for v in ovrlp_model._params.values() if v.requires_grad]

    return -2 * hessian(ovrlp, vparams)


def get_quantum_fisher_spsa(
    circuit: QuantumCircuit,
    iteration: int,
    vparams_values: Iterable[float | Tensor] | None = None,
    fm_dict: dict[str, Tensor] = {},
    previous_qfi_estimator: Tensor | None = None,
    epsilon: float = 10e-3,
    beta: float = 10e-2,
    backend: BackendName = BackendName.PYQTORCH,  # type: ignore
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,  # type: ignore
) -> Tensor:
    """Function to calculate the Quantum Fisher Information (QFI) matrix with the.

    SPSA approximation.

    Args:
        circuit (QuantumCircuit): The Quantum circuit we want to compute the QFI matrix of.
        iteration (int): Current number of iteration.
        vparams_values (tuple | list | Tensor | None):
            Values of the variational parameters where we want to compute the QFI.
        fm_dict (dict[str, Tensor]): Values of the feature map parameters.
        overla p_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.
    """
    # Get feature map dictionary (required to run Overlap().forward())
    if fm_dict == {}:
        fm_dict = _get_fm_dict(circuit)

    # Set variational parameters
    if vparams_values is not None:
        _set_circuit_vparams(circuit, vparams_values)

    # Get Overlap() model
    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    # Calculate the QFI matrix
    fid_hess = spsa_2gradient_step(ovrlp_model, epsilon, fm_dict)
    qfi_mat = -2 * fid_hess

    # Calculate the QFI estimator from the old estimator of qfi_mat
    if iteration == 0:
        qfi_mat_estimator = qfi_mat
    else:
        a_k = 1 / (1 + iteration)
        qfi_mat_estimator = a_k * (iteration * previous_qfi_estimator + qfi_mat)  # type: ignore

    # Get the positive-semidefinite version of the matrix for the update rule in QNG
    qfi_mat_positive_sd = _symsqrt(torch.matmul(qfi_mat_estimator, qfi_mat_estimator))
    qfi_mat_positive_sd = qfi_mat_positive_sd + beta * torch.eye(ovrlp_model.num_vparams)
    qfi_mat_positive_sd = qfi_mat_positive_sd / (1 + beta)  # regularization

    return qfi_mat_estimator, qfi_mat_positive_sd
