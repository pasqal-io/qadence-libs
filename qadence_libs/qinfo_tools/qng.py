from __future__ import annotations

from typing import Callable

import torch
from qadence import QuantumCircuit
from torch.optim.optimizer import Optimizer, required

from qadence_libs.qinfo_tools.qfi import get_quantum_fisher, get_quantum_fisher_spsa
from qadence_libs.types import FisherApproximation


class QuantumNaturalGradient(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm.

    There are currently two variants of the algorithm implemented: exact QNG and
    the SPSA approximation.

    WARNING: The exact QNG optimizer is very inefficient both in time and memory as
    it calculates the exact Quantum Fisher Information of the circuit at every
    iteration. Therefore, it is not meant to be run with medium to large circuits.
    Other approximations such as the SPSA are much more efficient while retaining
    good performance.
    """

    def __init__(
        self,
        params: tuple | torch.Tensor,
        circuit: QuantumCircuit = required,
        lr: float = required,
        approximation: FisherApproximation | str = FisherApproximation.SPSA,
        beta: float = 10e-3,
        epsilon: float = 10e-2,
    ):
        """
        Args:

            params (tuple | torch.Tensor): Variational parameters to be updated
            circuit (QuantumCircuit): Quantum circuit. Needed to compute the QFI matrix
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
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("The circuit should be an instance of qadence.QuantumCircuit")
        if 0.0 >= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if 0.0 > epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        if approximation == FisherApproximation.SPSA:
            self.iteration = 0
            self.prev_qfi_estimator = None

        defaults = dict(
            circuit=circuit,
            lr=lr,
            approximation=approximation,
            beta=beta,
            epsilon=epsilon,
        )
        super(QuantumNaturalGradient, self).__init__(params, defaults)

    def __setstate__(self, state):  # type: ignore
        super().__setstate__(state)

    def step(self, closure: Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step of the QNG algorithm.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            approximation = group["approximation"]
            grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])

            if approximation == FisherApproximation.EXACT:
                # Calculate the EXACT metric tensor
                metric_tensor = 0.25 * get_quantum_fisher(
                    group["circuit"],
                    vparams_values=group["params"],
                )

                with torch.no_grad():
                    # Finite shift the metric tensor to avoid problems when inverting
                    metric_tensor = metric_tensor + group["beta"] * torch.eye(len(grad_vec))

                    # Get transformed gradient vector
                    metric_tensor_inv = torch.linalg.inv(metric_tensor)
                    transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                    # Update parameters
                    for i, p in enumerate(group["params"]):
                        if p.grad is None:
                            continue
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

            elif approximation == FisherApproximation.SPSA:
                with torch.no_grad():
                    # Get estimation of the QFI matrix
                    qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
                        circuit=group["circuit"],
                        iteration=self.iteration,
                        vparams_values=group["params"],
                        previous_qfi_estimator=self.prev_qfi_estimator,
                        epsilon=group["epsilon"],
                        beta=group["beta"],
                    )

                    # Get transformed gradient vector
                    metric_tensor_inv = torch.linalg.pinv(0.25 * qfi_mat_positive_sd)
                    transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                    # Update parameters
                    for i, p in enumerate(group["params"]):
                        if p.grad is None:
                            continue
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

                self.iteration += 1
                self.prev_qfi_estimator = qfi_estimator

            else:
                raise NotImplementedError(
                    f"Approximation {approximation} of the QNG optimizer "
                    "is not implemented.Choose an item from the "
                    f"FisherApproximation enum: {FisherApproximation.list()},"
                )

        return loss
