from __future__ import annotations

import re
from typing import Callable

import torch
from qadence import QNN, QuantumCircuit, QuantumModel, Parameter
from qadence.logger import get_logger
from torch.optim.optimizer import Optimizer, required

from qadence_libs.qinfo_tools.qfi import get_quantum_fisher, get_quantum_fisher_spsa
from qadence_libs.types import FisherApproximation

logger = get_logger(__name__)


def _identify_circuit_vparams(
    model: QuantumModel | QNN, circuit: QuantumCircuit
) -> dict[str, Parameter]:
    """Returns the parameters of the model that are circuit parameters

     Args:
        model (QuantumModel|QNN): The model
        circuit (QuantumCircuit): The quantum circuit

    Returns:
        dict[str, Parameter]:
            Dictionary containing the circuit parameters
    """
    non_circuit_vparams = []
    circ_vparams = {}
    pattern = r"_params\."
    for n, p in model.named_parameters():
        n = re.sub(pattern, "", n)
        if p.requires_grad:
            print(n, p)
            print(type(p))
            if n in circuit.parameters():
                circ_vparams[n] = p
            else:
                non_circuit_vparams.append(n)

    if len(non_circuit_vparams) > 0:
        msg = f"""Parameters {non_circuit_vparams} are non-circuit trainable parameters.
                 Since the QNG optimizer can only optimize circuit parameters, these
                 parameter will not be optimized. Please use another optimizer for the
                 non-circuit parameters."""
        logger.warning(msg)

    return circ_vparams


class QuantumNaturalGradient(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm.

    There are currently two variants of the algorithm implemented: exact QNG and
    the SPSA approximation.

    Unlike other torch optimizers, QuantumNaturalGradient does not take a `Sequence`
    of parameters as an argument, but rather the QuantumModel whose parameters are to be
    optimized. All circuit parameters in the QuantumModel will be optimized.

    WARNING: The exact QNG optimizer is very inefficient both in time and memory as
    it calculates the exact Quantum Fisher Information of the circuit at every
    iteration. Therefore, it is not meant to be run with medium to large circuits.
    Other approximations such as the SPSA are much more efficient while retaining
    good performance.
    """

    def __init__(
        self,
        model: QuantumModel | QNN = required,
        lr: float = required,
        approximation: FisherApproximation | str = FisherApproximation.SPSA,
        beta: float = 10e-3,
        epsilon: float = 10e-2,
    ):
        """
        Args:
            model (QuantumModel):
                Model whose parameters are to be optimized
            lr (float): Learning rate.
            approximation (FisherApproximation):
                Approximation used to compute the QFI matrix. Defaults to FisherApproximation.SPSA
            beta (float):
                Shift applied to the QFI matrix before inversion to ensure numerical stability.
                Defaults to 10e-3.
            epsilon (float):
                Finite difference applied when computing the SPSA derivatives. Defaults to 10e-2.
        """

        if 0.0 > lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if 0.0 >= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if 0.0 > epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        if not isinstance(model, QuantumModel):
            raise TypeError(
                f"""The model should be an instance of '<class QuantumModel>'
                or '<class TransformedModule>'. Got {type(model)}."""
            )

        self.model = model
        self.circuit = model._circuit.abstract
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError(
                f"""The circuit should be an instance of '<class QuantumCircuit>'.
                Got {type(self.circuit)}"""
            )

        self._params_dict = _identify_circuit_vparams(model, self.circuit)
        params = list(self._params_dict.values())

        defaults = dict(
            lr=lr,
            approximation=approximation,
            beta=beta,
            epsilon=epsilon,
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("QNG doesn't support per-parameter options (parameter groups)")

        if approximation == FisherApproximation.SPSA:
            state = self.state
            state.setdefault("iter", 0)
            state.setdefault("qfi_estimator", None)

    def step(self, closure: Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step of the QNG algorithm.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        approximation = group["approximation"]
        beta = group["beta"]
        epsilon = group["epsilon"]
        lr = group["lr"]
        circuit = self.circuit
        vparams_dict = self._params_dict
        vparams = group["params"]
        grad_vec = torch.tensor([v.grad.data for v in vparams])

        if approximation == FisherApproximation.EXACT:
            qng_exact(vparams, grad_vec, vparams_dict, lr, circuit, beta)
        elif approximation == FisherApproximation.SPSA:
            qng_spsa(vparams, grad_vec, vparams_dict, lr, circuit, self.state, epsilon, beta)
        else:
            raise NotImplementedError(
                f"""Approximation {approximation} of the QNG optimizer
                is not implemented. Choose an item from the
                FisherApproximation enum: {FisherApproximation.list()}."""
            )

        return loss


def qng_exact(
    vparams: list,
    grad_vec: list,
    vparams_dict: dict,
    lr: float,
    circuit: QuantumCircuit,
    beta: float,
) -> None:
    """Functional API that performs exact QNG algorithm computation.

    See :class:`~qadence_libs.qinfo_tools.QuantumNaturalGradient` for details.
    """
    # EXACT metric tensor
    metric_tensor = 0.25 * get_quantum_fisher(
        circuit,
        vparams_dict=vparams_dict,
    )
    with torch.no_grad():
        # Apply a finite shift to the metric tensor to avoid numerical
        # stability issues when solving the least squares problem
        metric_tensor = metric_tensor + beta * torch.eye(len(grad_vec))

        # Get transformed gradient vector solving the least squares problem
        transf_grad = torch.linalg.lstsq(
            metric_tensor,
            grad_vec,
            driver="gelsd",
        ).solution

        # Update parameters
        for i, p in enumerate(vparams):
            p.data.add_(transf_grad[i], alpha=-lr)


def qng_spsa(
    vparams: list,
    grad_vec: list,
    vparams_dict: dict,
    lr: float,
    circuit: QuantumCircuit,
    state: dict,
    epsilon: float,
    beta: float,
) -> None:
    """Functional API that performs the QNG-SPSA algorithm computation.

    See :class:`~qadence_libs.qinfo_tools.QuantumNaturalGradient` for details.
    """
    # Get estimation of the QFI matrix
    qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
        circuit=circuit,
        iteration=state["iter"],
        vparams_dict=vparams_dict,
        previous_qfi_estimator=state["qfi_estimator"],
        epsilon=epsilon,
        beta=beta,
    )

    # Get transformed gradient vector solving the least squares problem
    transf_grad = torch.linalg.lstsq(
        0.25 * qfi_mat_positive_sd,
        grad_vec,
        driver="gelsd",
    ).solution

    # Update parameters
    for i, p in enumerate(vparams):
        if p.grad is None:
            continue
        p.data.add_(transf_grad[i], alpha=-lr)

    state["iter"] += 1
    state["qfi_estimator"] = qfi_estimator
