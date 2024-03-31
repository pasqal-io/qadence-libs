from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import grad


def hessian(output: Tensor, inputs: list) -> Tensor:
    """Calculates the Hessian of a given output vector wrt the inputs.

        TODO: Use autograd built-in functions for a more efficient implementation, but grad tree
    is broken by the Overlap method

        Args:
            output (Tensor): Output vector
            inputs (list): List of input parameters
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
