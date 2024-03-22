from __future__ import annotations

from typing import Callable

import torch
from torch.optim.optimizer import Optimizer, required
from qadence import QuantumCircuit
from qadence_libs.qinfo_tools.qfi import get_quantum_fisher, get_quantum_fisher_spsa


class QNG(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm."""

    def __init__(
        self,
        params: tuple | torch.Tensor,
        circuit: QuantumCircuit = required,
        lr: float = required,
        beta: float = 10e-3,
    ):
        defaults = dict(circuit=circuit, lr=lr, beta=beta)
        super(QNG, self).__init__(params, defaults)

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
            grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])

            # Calculate the EXACT metric tensor
            metric_tensor = (1 / 4) * get_quantum_fisher(group["circuit"])

            with torch.no_grad():

                # Finite shift the metric tensor to avoid problems when inverting
                metric_tensor += group["beta"] * torch.eye(len(grad_vec))

                # Invert matrix
                metric_tensor_inv = torch.linalg.pinv(metric_tensor)

                # Calculate the tranformed gradient vector
                transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                # Update parameters
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    p.data.add_(transf_grad[i], alpha=-group["lr"])

        return loss


class QNG_SPSA(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm using the SPSA approximation
    to iteratively construct an approximation of the Quantum Fisher Information."""

    def __init__(
        self,
        params: tuple | torch.Tensor,
        circuit: QuantumCircuit = required,
        lr: float = required,
        iteration: int = 0,
        beta: float = 10e-3,
        epsilon: float = 10e-2,
    ):
        self.iteration = iteration
        self.prev_qfi_estimator = 0
        defaults = dict(
            circuit=circuit,
            lr=lr,
            epsilon=epsilon,
            beta=beta,
        )
        super(QNG_SPSA, self).__init__(params, defaults)

    def __setstate__(self, state):  # type: ignore
        super().__setstate__(state)

    def step(self, closure: Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step of the QNG-SPSA algorithm.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])

            with torch.no_grad():
                # Get estimation of the QFI matrix
                qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
                    circuit=group["circuit"],
                    k=self.iteration,
                    vparams_values=group["params"],
                    previous_qfi_estimator=self.prev_qfi_estimator,
                    epsilon=group["epsilon"],
                    beta=group["beta"],
                )

                metric_tensor = (1 / 4) * qfi_mat_positive_sd
                metric_tensor_inv = torch.linalg.pinv(metric_tensor)
                transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                # Update parameters
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    p.data.add_(transf_grad[i], alpha=-group["lr"])

            self.iteration += 1
            self.prev_qfi_estimator = qfi_estimator

        return loss
