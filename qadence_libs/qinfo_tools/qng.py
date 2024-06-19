from __future__ import annotations

from typing import Callable, Sequence

import torch
from qadence import QuantumCircuit, QuantumModel
from qadence.logger import get_logger
from qadence.ml_tools.models import TransformedModule
from torch.optim.optimizer import Optimizer, required

from qadence_libs.qinfo_tools.qfi import get_quantum_fisher, get_quantum_fisher_spsa
from qadence_libs.types import FisherApproximation

logger = get_logger(__name__)


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
        params: Sequence,
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
        if 0.0 >= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if 0.0 > epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        if isinstance(model, TransformedModule):
            logger.warning(
                "The model is of type '<class TransformedModule>. "
                "Keep in mind that the QNG optimizer can only optimize circuit "
                "parameters. Input and output shifting/scaling parameters will not be optimized."
            )
            # Retrieve the quantum model from the TransformedModule
            model = model.model
        if not isinstance(model, QuantumModel):
            raise TypeError(
                "The model should be an instance of '<class QuantumModel>' "
                f"or '<class TransformedModule>'. Got {type(model)}."
            )

        self.model = model
        self.circuit = model._circuit.abstract
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError(
                "The circuit should be an instance of '<class QuantumCircuit>'."
                "Got {type(self.circuit)}"
            )

        defaults = dict(
            lr=lr,
            approximation=approximation,
            beta=beta,
            epsilon=epsilon,
        )

        super().__init__(params, defaults)

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

        for group in self.param_groups:
            # Parameters passed to the optimizer
            vparams_values = [p for p in group["params"] if p.requires_grad]

            # Build the parameter dictionary
            # We rely on the `vparam()` method in `QuantumModel` and the
            # `parameters()` in `nn.Module` to give the same param ordering.
            # We test for this in `test_qng.py`.
            vparams_dict = dict(zip(self.model.vparams.keys(), vparams_values))

            approximation = group["approximation"]
            grad_vec = torch.tensor([v.grad.data for v in vparams_values])
            if approximation == FisherApproximation.EXACT:
                # Calculate the EXACT metric tensor
                metric_tensor = 0.25 * get_quantum_fisher(
                    self.circuit,
                    vparams_dict=vparams_dict,
                )

                with torch.no_grad():
                    # Apply a finite shift to the metric tensor to avoid numerical
                    # stability issues when solving the least squares problem
                    metric_tensor = metric_tensor + group["beta"] * torch.eye(len(grad_vec))

                    # Get transformed gradient vector solving the least squares problem
                    transf_grad = torch.linalg.lstsq(
                        metric_tensor,
                        grad_vec,
                        driver="gelsd",
                    ).solution

                    # Update parameters
                    for i, p in enumerate(vparams_values):
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

            elif approximation == FisherApproximation.SPSA:
                state = self.state
                with torch.no_grad():
                    # Get estimation of the QFI matrix
                    qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
                        circuit=self.circuit,
                        iteration=state["iter"],
                        vparams_dict=vparams_dict,
                        previous_qfi_estimator=state["qfi_estimator"],
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
                    for i, p in enumerate(vparams_values):
                        if p.grad is None:
                            continue
                        p.data.add_(transf_grad[i], alpha=-group["lr"])

                state["iter"] += 1
                state["qfi_estimator"] = qfi_estimator

            else:
                raise NotImplementedError(
                    f"Approximation {approximation} of the QNG optimizer "
                    "is not implemented. Choose an item from the "
                    f"FisherApproximation enum: {FisherApproximation.list()}."
                )

        return loss
