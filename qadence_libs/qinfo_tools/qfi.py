from __future__ import annotations

import torch
from qadence import Overlap, QuantumCircuit
from qadence.blocks import parameters, primitive_blocks
from qadence.types import BackendName, DiffMode, OverlapMethod
from torch import Tensor

from qadence_libs.qinfo_tools.spsa import spsa_2gradient_step
from qadence_libs.qinfo_tools.utils import hessian


def _positive_semidefinite_sqrt(A: Tensor) -> Tensor:
    """Computes the square root of a real positive semi-definite matrix.

    Code taken from https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def _set_circuit_vparams(circuit: QuantumCircuit, vparams_dict: dict[str, Tensor]) -> None:
    """Sets the variational parameter values of the circuit."""
    blocks = primitive_blocks(circuit.block)
    for block in blocks:
        params = parameters(block)
        for p in params:
            if p.trainable:
                p.value = vparams_dict[p.name]


def _get_fm_dict(circuit: QuantumCircuit) -> dict[str, Tensor]:
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
    vparams_dict: dict[str, Tensor] = dict(),
    fm_dict: dict[str, Tensor] = dict(),
    backend: BackendName = BackendName.PYQTORCH,
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,
) -> Tensor:
    """Returns the exact Quantum Fisher Information (QFI) matrix.

    Args:
        circuit (QuantumCircuit): The quantum circuit.
        vparams_dict (dict[str, Tensor]):
            Dictionary holding the values of the variational parameters of the circuit.
        fm_dict (dict[str, Tensor]):
            Dictionary holding the values of the Feature Map parameters of the circuit.
        overlap_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.

    Returns:
        Tensor:
            The exact QFI matrix of the circuit.
    """
    # The FM dictionary is needed for the forward method in Overlap()
    if not fm_dict:
        fm_dict = _get_fm_dict(circuit)

    if vparams_dict:
        _set_circuit_vparams(circuit, vparams_dict)

    overlap_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )
    overlap = overlap_model(bra_param_values=fm_dict, ket_param_values=fm_dict)

    # Retrieve variational parameters of the overlap model
    # Importantly, the vparams of the overlap model are the vparams of the bra tensor,
    # Which means if we differentiate wrt vparams we are differentiating only wrt the
    # parameters in the bra and not in the ket
    vparams = [v for v in overlap_model._params.values() if v.requires_grad]
    return -2 * hessian(overlap, vparams)


def get_quantum_fisher_spsa(
    circuit: QuantumCircuit,
    iteration: int,
    vparams_dict: dict[str, Tensor] = dict(),
    fm_dict: dict[str, Tensor] = dict(),
    previous_qfi_estimator: Tensor | None = None,
    epsilon: float = 10e-3,
    beta: float = 10e-2,
    backend: BackendName = BackendName.PYQTORCH,
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,
) -> tuple[Tensor, Tensor]:
    """Returns the a SPSA-approximation of the Quantum Fisher Information (QFI) matrix.

    Args:
        circuit (QuantumCircuit): The quantum circuit.
        iteration (int): Current iteration in the SPSA iterative loop.
        vparams_values (dict[str, Tensor]):
            Dictionary holding the values of the variational parameters of the circuit.
        fm_dict (dict[str, Tensor]):
            Dictionary holding the values of the Feature Map parameters of the circuit.
        overla p_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.

    Returns:
        tuple[Tensor, Tensor]:
            Tuple containing the QFI matrix and its positive semi-definite estimator.
    """
    # Retrieve feature parameters if they are not given as inputs
    # The FM dictionary is needed for the forward run of the Overlap() model
    if not fm_dict:
        fm_dict = _get_fm_dict(circuit)

    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    # Calculate the QFI matrix
    qfi_mat = -2 * spsa_2gradient_step(ovrlp_model, epsilon, fm_dict, vparams_dict)

    # Calculate the QFI estimator from the old estimator of qfi_mat
    if iteration == 0:
        qfi_mat_estimator = qfi_mat
    else:
        a_k = 1 / (1 + iteration)
        qfi_mat_estimator = a_k * (iteration * previous_qfi_estimator + qfi_mat)  # type: ignore

    # Get the positive-semidefinite version of the matrix for the update rule in QNG
    qfi_mat_positive_sd = _positive_semidefinite_sqrt(qfi_mat_estimator @ qfi_mat_estimator)
    qfi_mat_positive_sd = qfi_mat_positive_sd + beta * torch.eye(ovrlp_model.num_vparams)
    qfi_mat_positive_sd = qfi_mat_positive_sd / (1 + beta)  # regularization

    return qfi_mat_estimator, qfi_mat_positive_sd
