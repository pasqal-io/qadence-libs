from __future__ import annotations

from typing import Callable

import torch
from qadence import QuantumCircuit
from torch.optim.optimizer import Optimizer, required

from qadence_libs.qinfo_tools.qfi import get_quantum_fisher, get_quantum_fisher_spsa


class QNG(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm.

    WARNING: This class implements the exact QNG optimizer, which is very inefficient
    both in time and memory as it calculates the exact Quantum Fisher Information of
    the circuit at every iteration. Therefore, it is not meant to be run with medium
    to large circuits. Other approximations such as the QNG-SPSA optimizer are much
    more efficient while retaining good performance.
    """

    def __init__(
        self,
        params: tuple | torch.Tensor,
        circuit: QuantumCircuit = required,
        lr: float = required,
        beta: float = 10e-3,
    ):
        """
        Args:

            params (tuple | torch.Tensor): Variational parameters to be updated
            circuit (QuantumCircuit): Quantum circuit. Needed to compute the QFI matrix
            lr (float): Learning rate.
            beta (float):
                Shift applied to the QFI matrix before inversion to ensure numerical stability.
                Defaults to 10e-3.
        """

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"The circuit should be an instance of {type(QuantumCircuit)}")

        defaults = dict(circuit=circuit, lr=lr, beta=beta)
        super(QNG, self).__init__(params, defaults)

    def __setstate__(self, state):  # type: ignore
        super().__setstate__(state)

    def step(self, closure: Callable | None = None) -> Tensor:
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

        return loss


class QNG_SPSA(Optimizer):
    """
    Implements the Quantum Natural Gradient Algorithm using the SPSA algorithm.

    In the QNG-SPSA algorithm, the SPSA algorithm is used to iteratively construct
    an approximation of the Quantum Fisher Information matrix, which is then used
    in the parameter update rule of the optimizer.
    """

    def __init__(
        self,
        params: tuple | torch.Tensor,
        circuit: QuantumCircuit = required,
        lr: float = required,
        iteration: int = 0,
        beta: float = 10e-3,
        epsilon: float = 10e-2,
    ):
        """
        Args:

            params (tuple | torch.Tensor): Variational parameters to be updated
            circuit (QuantumCircuit): Quantum circuit. Required to compute the QFI matrix.
            lr (float): Learning rate.
            iteration (int): Current iteration. Required to compute the SPSA estimator of the QFI.
            beta (float):
                Shift applied to the QFI matrix before inversion to ensure numerical stability.
                Defaults to 10e-3.
            epsilon (float):
                Finite shift applied when computing the SPSA derivatives. Defaults to 10e-2.
        """

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("The circuit should be an instance of qadence.QuantumCircuit")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        self.current_iteration = iteration
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
                    iteration=self.current_iteration,
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

            self.current_iteration += 1
            self.prev_qfi_estimator = qfi_estimator

        return loss
