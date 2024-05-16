from __future__ import annotations

from typing import Callable

import torch
from qadence import QuantumCircuit, QuantumModel
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
        model: QuantumModel = required,
        lr: float = required,
        approximation: FisherApproximation | str = FisherApproximation.SPSA,
        beta: float = 10e-3,
        epsilon: float = 10e-2,
    ):
        """
        Args:

            params (tuple | torch.Tensor): Variational parameters to be updated
            model (QuantumModel):
                Model to be optimized. The optimizers needs to access its quantum circuit
                to compute the QFI matrix.
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
        if not isinstance(model, QuantumModel):
            raise ValueError(
                f"The model should be an instance of '<class QuantumModel>'. Got {type(model)}"
            )
        if 0.0 >= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if 0.0 > epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        if approximation == FisherApproximation.SPSA:
            self.iteration = 0
            self.prev_qfi_estimator = None

        self.circuit = model._circuit.abstract
        if not isinstance(self.circuit, QuantumCircuit):
            raise ValueError(
                f"The circuit should be an instance of '<class QuantumCircuit>'. Got {type(self.circuit)}"
            )

        defaults = dict(
            model=model,
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

            vparams = [p for p in group["params"] if p.requires_grad]
            approximation = group["approximation"]
            grad_vec = torch.tensor([v.grad.data for v in vparams])

            if approximation == FisherApproximation.EXACT:
                # Calculate the EXACT metric tensor
                metric_tensor = 0.25 * get_quantum_fisher(
                    self.circuit,
                    vparams_values=vparams,
                )

                with torch.no_grad():
                    # Finite shift the metric tensor to avoid problems when inverting
                    metric_tensor = metric_tensor + group["beta"] * torch.eye(len(grad_vec))

                    # Get transformed gradient vector solving the least squares problem
                    transf_grad = torch.linalg.lstsq(
                        metric_tensor,
                        grad_vec,
                        driver="gelsd",
                    ).solution

                    # Update parameters
                    for i, p in enumerate(vparams):
                        if p.grad is None:
                            continue
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

            elif approximation == FisherApproximation.SPSA:
                with torch.no_grad():
                    # Get estimation of the QFI matrix
                    qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
                        circuit=self.circuit,
                        iteration=self.iteration,
                        vparams_values=vparams,
                        previous_qfi_estimator=self.prev_qfi_estimator,
                        epsilon=group["epsilon"],
                        beta=group["beta"],
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
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

                self.iteration += 1
                self.prev_qfi_estimator = qfi_estimator

            else:
                raise NotImplementedError(
                    f"Approximation {approximation} of the QNG optimizer "
                    "is not implemented. Choose an item from the "
                    f"FisherApproximation enum: {FisherApproximation.list()},"
                )

        return loss
