from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import grad

from qadence import Overlap
from qadence.types import OverlapMethod, BackendName, DiffMode
from qadence.blocks import parameters, primitive_blocks
from qadence.circuit import QuantumCircuit

from qadence_libs.qinfo_tools.spsa import spsa_2gradient


def _symsqrt(A):
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices.
    Code from https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228"""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def _set_circuit_vparams(circuit, vparams_values):
    if vparams_values is not None:
        blocks = primitive_blocks(circuit.block)
        for i, block in enumerate(blocks):
            params = parameters(block)
            for p in params:
                if p.trainable:
                    p.value = float(vparams_values[i])


def _set_fm_dict(fm_dict, circuit):
    blocks = primitive_blocks(circuit.block)
    for block in blocks:
        params = parameters(block)
        for p in params:
            if not p.trainable:
                fm_dict[p.name] = torch.tensor([p.value])


def hessian(output: Tensor, inputs: list) -> Tensor:
    """Calculates the Hessian of a given output vector wrt the inputs.

    This is not an efficient method. Probably better to use autograd functions but grad tree is broken by the Overlap method

    Args:
        output (Tensor): _description_
        inputs (list): _description_

    Returns:
        Tensor: _description_
    """

    jacobian = grad(
        output,
        inputs,
        torch.ones_like(output),
        create_graph=True,
        allow_unused=True,
    )

    n_params = len(inputs)
    hess = torch.empty((n_params, n_params))
    for i in range(n_params):
        ovrlp_grad2 = grad(
            jacobian[i],
            inputs,
            torch.ones_like(jacobian[i]),
            create_graph=True,
            allow_unused=True,
        )
        for j in range(n_params):
            hess[i, j] = ovrlp_grad2[j]

    return hess


def get_quantum_fisher(
    circuit: QuantumCircuit,
    vparams_values: tuple | list | Tensor | None = None,
    fm_dict: dict[str, Tensor] = {},
    backend: BackendName = BackendName.PYQTORCH,  # type: ignore
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,  # type: ignore
) -> Tensor:
    """Function to calculate the exact Quantum Fisher Information (QFI) matrix of the
    quantum circuit with given values for the variational parameters (vparams_values) and the
    feature map (fm_dict).

    Args:
        circuit (QuantumCircuit): The Quantum circuit we want to compute the QFI matrix of.
        vparams (tuple): Values of the variational parameters where we want to compute the QFI.
        fm_dict (dict[str, Tensor]): Values of the feature map parameters.
        overlap_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.

    Returns:
        torch.Tensor: QFI matrix
    """

    # If fm_dict is not provided, build the feature map dictionary
    if fm_dict == {}:
        _set_fm_dict(fm_dict, circuit)

    # Get Overlap() model
    _set_circuit_vparams(circuit, vparams_values)
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

    # Calculate the Hessian
    fid_hess = hessian(ovrlp, vparams)

    return -2 * fid_hess


def get_quantum_fisher_spsa(
    circuit: QuantumCircuit,
    k: int,
    vparams_values: tuple | list | Tensor | None = None,
    fm_dict: dict[str, Tensor] = {},
    previous_qfi_estimator: Tensor = None,
    epsilon: float = 10e-4,
    beta: float = 10e-3,
    backend: BackendName = BackendName.PYQTORCH,  # type: ignore
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,  # type: ignore
) -> Tensor:
    """Function to calculate the Quantum Fisher Information (QFI) matrix with the
    SPSA approximation.

    Args:
        circuit (QuantumCircuit): The Quantum circuit we want to compute the QFI matrix of.
        k (int): Current number of iteration.
        vparams_values (tuple): Values of the variational parameters where we want to compute the QFI.
        fm_dict (dict[str, Tensor]): Values of the feature map parameters.
        overla p_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.
    Returns:
        torch.Tensor: QFI matrix
    """

    # If fm_dict is not provided, build the feature map dictionary
    if fm_dict == {}:
        _set_fm_dict(fm_dict, circuit)

    # Hessian of the overlap model via SPSA
    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )
    fid_hess = spsa_2gradient(ovrlp_model, epsilon, fm_dict, vparams_values)

    # Quantum Fisher Information matrix
    qfi_mat = -2 * fid_hess

    # Calculate the new estimator from the old estimator of qfi_mat
    if k == 0:
        qfi_mat_estimator = qfi_mat
    else:
        qfi_mat_estimator = (1 / (k + 1)) * (k * previous_qfi_estimator + qfi_mat)  # type: ignore

    # Get the positive-semidefinite version of the matrix for the update rule in QNG
    qfi_mat_positive_sd = _symsqrt(torch.matmul(qfi_mat_estimator, qfi_mat_estimator))
    qfi_mat_positive_sd = qfi_mat_positive_sd + beta * torch.eye(len(vparams_values))

    return qfi_mat_estimator, qfi_mat_positive_sd


if __name__ == "__main__":

    import torch
    import time
    import numpy as np

    from qadence.constructors import hea, feature_map
    from qadence.operations import *
    from qadence import QuantumCircuit

    torch.manual_seed(0)
    np.random.seed(0)

    torch.set_printoptions(precision=3, sci_mode=False)

    n_qubits = 2
    batch_size = 1
    layers = 1
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type="fourier")
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX, RY])
    circuit = QuantumCircuit(n_qubits, ansatz, fm)

    # Decide which values of the parameters we want to calculate the QFI in
    vparams = [vparam for vparam in circuit.parameters() if vparam.trainable]
    var_values = tuple(torch.rand(1, requires_grad=True) for i in range(len(vparams)))
    var_values_nograd = [var_values[i].detach().numpy()[0] for i in range(len(var_values))]

    print("qadence QFI exact")
    fm_dict = {"phi": torch.Tensor([0])}
    t0 = time.time()
    qfi_mat_exact = get_quantum_fisher(circuit, vparams_values=var_values, fm_dict=fm_dict)
    t1 = time.time()
    total = t1 - t0
    print(f"Total time {total}")
    print(qfi_mat_exact)

    print("qadence QFI with SPSA approximation")
    qfi_mat_spsa = None
    t0 = time.time()
    for k in range(50):
        qfi_mat_spsa, qfi_positive_sd = get_quantum_fisher_spsa(
            circuit,
            k,
            vparams_values=var_values,
            fm_dict=fm_dict,
            previous_qfi_estimator=qfi_mat_spsa,
        )
    t1 = time.time()
    total = t1 - t0
    print(f"Total time {total}")

    print(qfi_mat_spsa)
