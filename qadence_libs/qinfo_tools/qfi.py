from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad

from qadence import Overlap
from qadence.types import OverlapMethod, BackendName, DiffMode
from qadence.blocks import parameters, primitive_blocks
from qadence.circuit import QuantumCircuit


def symsqrt(A):
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices"""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


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
    if fm_dict == {}:
        blocks = primitive_blocks(circuit.block)
        for block in blocks:
            params = parameters(block)
            for p in params:
                if not p.trainable:
                    fm_dict[p.name] = torch.tensor([p.value])

    # Get Overlap() model
    ovrlp_model = _overlap_with_ket_inputs(
        circuit,
        vparams_values,  # type: ignore [arg-type]
        backend=backend,
        overlap_method=overlap_method,
        diff_mode=diff_mode,
    )
    ovrlp = ovrlp_model(bra_param_values=fm_dict, ket_param_values=fm_dict)

    # Retrieve variational parameters of the overlap model
    # Importantly, the vparams of the overlap model are the vparams of the bra tensor,
    # Which means if we differentiate wrt vparams we are differentiating only wrt the
    # parameters in the bra and not in the ket
    vparams = {k: v for k, v in ovrlp_model._params.items() if v.requires_grad}

    # Jacobian of the overlap
    ovrlp_grad = grad(
        ovrlp,
        list(vparams.values()),
        torch.ones_like(ovrlp),
        create_graph=True,
        allow_unused=True,
    )

    # Hessian of the overlap = QFI matrix
    n_params = ovrlp_model.num_vparams
    fid_hess = torch.empty((n_params, n_params))
    for i in range(n_params):
        ovrlp_grad2 = grad(
            ovrlp_grad[i],
            list(vparams.values()),
            torch.ones_like(ovrlp_grad[i]),
            create_graph=True,
            # allow_unused=,
        )
        for j in range(n_params):
            fid_hess[i, j] = ovrlp_grad2[j]

    return -2 * fid_hess


def _overlap_with_ket_inputs(
    circuit: QuantumCircuit,
    vparams_values: tuple | list | Tensor | None = None,
    backend: BackendName = BackendName.PYQTORCH,  # type: ignore
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,  # type: ignore
) -> Overlap:
    """Builds an OverlapModel consisting of the overlap of the circuit with itself.

    The variational parameters of the output model correspond to the variational parameters
    of the bra state, so we can do the differentiation properly.

    Args:
        circuit (QuantumCircuit): The Quantum circuit we want to compute the overlap of.
        vparams (tuple): Values of the variational parameters.
        backend (BackendName, optional): Defaults to BackendName.pyq.
        overlap_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): Defaults to DiffMode.ad.

    Returns:
        Overlap(): _description_
    """
    if vparams_values is not None:
        blocks = primitive_blocks(circuit.block)
        iter_index = iter(range(len(blocks)))
        for block in blocks:
            params = parameters(block)
            for p in params:
                if p.trainable:
                    p.value = float(vparams_values[next(iter_index)])

    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    return ovrlp_model


# def get_quantum_fisher(
#     circuit: QuantumCircuit,
#     vparams_values: tuple | list | Tensor | None = None,
#     fm_dict: dict[str, Tensor] = {},
#     backend: BackendName = BackendName.PYQTORCH,  # type: ignore
#     overlap_method: OverlapMethod = OverlapMethod.EXACT,
#     diff_mode: DiffMode = DiffMode.AD,  # type: ignore
# ) -> Tensor:
#     """Function to calculate the exact Quantum Fisher Information (QFI) matrix of the
#     quantum circuit with given values for the variational parameters (vparams_values) and the
#     feature map (fm_dict).

#     Args:
#         circuit (QuantumCircuit): The Quantum circuit we want to compute the QFI matrix of.
#         vparams (tuple): Values of the variational parameters where we want to compute the QFI.
#         fm_dict (dict[str, Tensor]): Values of the feature map parameters.
#         overlap_method (OverlapMethod, optional): Defaults to OverlapMethod.EXACT.
#         diff_mode (DiffMode, optional): Defaults to DiffMode.ad.

#     Returns:
#         torch.Tensor: QFI matrix
#     """
#     if fm_dict == {}:
#         blocks = primitive_blocks(circuit.block)
#         for block in blocks:
#             params = parameters(block)
#             for p in params:
#                 if not p.trainable:
#                     fm_dict[p.name] = torch.tensor([p.value])

#     # Get Overlap() model
#     def overlap_wrapper(vparams_values):
#         ovrlp_model = _overlap_with_ket_inputs(
#             circuit,
#             vparams_values,
#             backend=backend,
#             overlap_method=overlap_method,
#             diff_mode=diff_mode,
#         )
#         return ovrlp_model(bra_param_values=fm_dict, ket_param_values=fm_dict)

#     # print("overlap_result")
#     # print(ovrlp)

#     # def overlap_wrapper(vparams_values):
#     #     ovrlp()

#     # print(vparams_values)
#     # hess = torch.func.hessian(overlap_wrapper)(vparams_values)
#     # print(hess)

#     # Retrieve variational parameters of the overlap model
#     # Importantly, the vparams of the overlap model are the vparams of the bra tensor,
#     # Which means if we differentiate wrt vparams we are differentiating only wrt the
#     # parameters in the bra and not in the ket
#     vparams = {k: v for k, v in ovrlp_model._params.items() if v.requires_grad}

#     # Jacobian of the overlap
#     ovrlp_grad = grad(
#         ovrlp,
#         list(vparams.values()),
#         torch.ones_like(ovrlp),
#         create_graph=True,
#         allow_unused=True,
#     )

#     # Hessian of the overlap = QFI matrix
#     n_params = ovrlp_model.num_vparams
#     fid_hess = torch.empty((n_params, n_params))
#     for i in range(n_params):
#         ovrlp_grad2 = grad(
#             ovrlp_grad[i],
#             list(vparams.values()),
#             torch.ones_like(ovrlp_grad[i]),
#             create_graph=True,
#             allow_unused=True,
#         )
#         for j in range(n_params):
#             fid_hess[i, j] = ovrlp_grad2[j]

#     return -2 * fid_hess


class OverlapGradientSPSA(Overlap):
    def __init__(
        self,
        bra_circuit,
        ket_circuit,
        epsilon,
        vparams_values,
        fm_dict,
        backend,
        diff_mode,
        method,
    ):
        super().__init__(
            bra_circuit=bra_circuit,
            ket_circuit=ket_circuit,
            backend=backend,
            diff_mode=diff_mode,
            method=method,
        )
        self.epsilon = epsilon
        self.fm_dict = fm_dict
        self.vparams_values = vparams_values
        self.vparams_dict = {k: v for (k, v) in self._params.items() if v.requires_grad}
        self.vparams_tensors = torch.Tensor(vparams_values).reshape((self.num_vparams, 1))

    def _shifted_ket_vparam_dict(self, shift):
        vparams = {k: v for (k, v) in self._params.items() if v.requires_grad}
        return dict(zip(vparams.keys(), self.vparams_tensors + shift))

    def _shifted_overlap(self, shifted_vparams_dict):
        ovrlp_shifted = super().forward(
            bra_param_values=self.fm_dict | self.vparams_dict,
            ket_param_values=self.fm_dict | shifted_vparams_dict,
        )
        return ovrlp_shifted

    def _create_random_direction(self):
        return torch.Tensor(np.random.choice([-1, 1], size=(self.num_vparams, 1)))

    def first_order_gradient(self):
        # Create random direction
        random_direction = self._create_random_direction()

        # Shift ket variational parameters
        vparams_plus = self._shifted_ket_vparam_dict(self.epsilon * random_direction)
        vparams_minus = self._shifted_ket_vparam_dict(-self.epsilon * random_direction)

        # Overlaps with the shifted parameters
        ovrlp_shifter_plus = self._shifted_overlap(vparams_plus)
        ovrlp_shifter_minus = self._shifted_overlap(vparams_minus)

        return (ovrlp_shifter_plus - ovrlp_shifter_minus) / (2 * self.epsilon)

    def second_order_gradient(self):
        # Create random directions
        rand_dir1 = self._create_random_direction()
        rand_dir2 = self._create_random_direction()

        # Shift ket variational parameters
        vparams_p1 = self._shifted_ket_vparam_dict(self.epsilon * rand_dir1)
        vparams_p1p2 = self._shifted_ket_vparam_dict(self.epsilon * (rand_dir1 + rand_dir2))
        vparams_m1 = self._shifted_ket_vparam_dict(-self.epsilon * rand_dir1)
        vparams_m1p2 = self._shifted_ket_vparam_dict(self.epsilon * (-rand_dir1 + rand_dir2))

        # Overlaps with the shifted parameters
        ovrlp_shifted_p1 = self._shifted_overlap(vparams_p1)
        ovrlp_shifted_p1p2 = self._shifted_overlap(vparams_p1p2)
        ovrlp_shifted_m1 = self._shifted_overlap(vparams_m1)
        ovrlp_shifted_m1p2 = self._shifted_overlap(vparams_m1p2)

        # Prefactor
        delta_F = ovrlp_shifted_p1p2 - ovrlp_shifted_p1 - ovrlp_shifted_m1p2 + ovrlp_shifted_m1

        # Hessian
        hess = (
            (1 / 4)
            * (delta_F / (self.epsilon**2))
            * (
                torch.matmul(rand_dir1, rand_dir2.transpose(0, 1))
                + torch.matmul(rand_dir2, rand_dir1.transpose(0, 1))
            )
        )
        return hess


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
    num_vparams = len(vparams_values) if vparams_values is not None else 0

    if fm_dict == {}:
        blocks = primitive_blocks(circuit.block)
        for block in blocks:
            params = parameters(block)
            for p in params:
                if not p.trainable:
                    fm_dict[p.name] = torch.Tensor([p.value])

    # Hessian of the overlap model via SPSA
    spsa_gradient_model = OverlapGradientSPSA(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
        epsilon=epsilon,
        vparams_values=vparams_values,
        fm_dict=fm_dict,
    )
    fid_hess = spsa_gradient_model.second_order_gradient()

    # Quantum Fisher Information matrix
    qfi_mat = -2 * fid_hess

    # Calculate the new estimator from the old estimator of qfi_mat
    if k == 0:
        qfi_mat_estimator = qfi_mat
    else:
        qfi_mat_estimator = (1 / (k + 1)) * (k * previous_qfi_estimator + qfi_mat)  # type: ignore

    # Get the positive-semidefinite version of the matrix for the update rule in QNG
    qfi_mat_positive_sd = symsqrt(torch.matmul(qfi_mat_estimator, qfi_mat_estimator))
    qfi_mat_positive_sd = qfi_mat_positive_sd + beta * torch.eye(num_vparams)

    return qfi_mat_estimator, qfi_mat_positive_sd


if __name__ == "__main__":

    import torch
    import numpy as np

    from qadence.constructors import hea, feature_map
    from qadence.operations import *
    from qadence import QuantumCircuit

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
    qfi_mat_exact = get_quantum_fisher(circuit, vparams_values=var_values, fm_dict=fm_dict)
    print(qfi_mat_exact)

    print("qadence QFI with SPSA approximation")
    qfi_mat_spsa = None
    for k in range(1000):
        qfi_mat_spsa, qfi_positive_sd = get_quantum_fisher_spsa(
            circuit,
            k,
            vparams_values=var_values,
            fm_dict=fm_dict,
            previous_qfi_estimator=qfi_mat_spsa,
        )
    print(qfi_mat_spsa)
