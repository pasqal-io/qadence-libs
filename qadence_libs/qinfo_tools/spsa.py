from __future__ import annotations

import numpy as np
import torch
from qadence import Overlap
from torch import Tensor


def _create_random_direction(size: int) -> Tensor:
    """
    Creates a torch Tensor with elements randomly drawn from {-1,+1}.

    Args:
        size (int): Size of the vector
    """
    return torch.Tensor(np.random.choice([-1, 1], size=(size, 1)))


def _shifted_overlap(
    model: Overlap,
    shift: Tensor,
    fm_dict: dict,
    vparams_dict: dict,
) -> Tensor:
    """
    Calls the forward method of the model with shifted values of the ket variational parameters.

    Args:
        model (Overlap): Overlap model
        shift (float): Quantity of the shift
        fm_dict (dict): Feature map dictionary
        vparams_dict (dict): Variational parameter dictionary
    """
    shifted_vparams_dict = {k: (v + s) for (k, v), s in zip(vparams_dict.items(), shift)}

    ovrlp_shifted = model(
        bra_param_values=fm_dict | vparams_dict,
        ket_param_values=fm_dict | shifted_vparams_dict,
    )
    return ovrlp_shifted


def spsa_gradient_step(
    model: Overlap,
    epsilon: float,
    fm_dict: dict,
) -> Tensor:
    """
    Single step to calculate the first order gradient of the given Overlap model.

    via the SPSA approximation.

    Args:
        model (Overlap): Overlap model
        epsilon (float): Finite step size
        fm_dict (dict | None): Feature map dictionary
    """

    vparams_dict = {k: v for (k, v) in model._params.items() if v.requires_grad}

    # Create random direction
    random_direction = _create_random_direction(size=model.num_vparams)

    # Shift ket variational parameters
    shift = epsilon * random_direction

    # Overlaps with the shifted parameters
    ovrlp_shifted_plus = _shifted_overlap(model, shift, fm_dict, vparams_dict)
    ovrlp_shifted_minus = _shifted_overlap(model, -shift, fm_dict, vparams_dict)

    return random_direction * (ovrlp_shifted_plus - ovrlp_shifted_minus) / (2 * epsilon)


def spsa_2gradient_step(
    model: Overlap,
    epsilon: float,
    fm_dict: dict,
) -> Tensor:
    """
    Single step to calculate the second order gradient of the given Overlap model.

    via the SPSA approximation.

    TODO: implement recursively using the first order function

    Args:
        model (Overlap): Overlap model
        epsilon (float): Finite step size
        fm_dict (dict | None): Feature map dictionary
    """
    vparams_dict = {k: v for (k, v) in model._params.items() if v.requires_grad}

    # Create random directions
    rand_dir1 = _create_random_direction(size=model.num_vparams)
    rand_dir2 = _create_random_direction(size=model.num_vparams)

    # Overlaps with the shifted parameters
    shift_p1 = epsilon * rand_dir1
    ovrlp_shifted_p1 = _shifted_overlap(model, shift_p1, fm_dict, vparams_dict)
    shift_p1p2 = epsilon * (rand_dir1 + rand_dir2)
    ovrlp_shifted_p1p2 = _shifted_overlap(model, shift_p1p2, fm_dict, vparams_dict)
    shift_m1 = -epsilon * rand_dir1
    ovrlp_shifted_m1 = _shifted_overlap(model, shift_m1, fm_dict, vparams_dict)
    shift_m1p2 = epsilon * (-rand_dir1 + rand_dir2)
    ovrlp_shifted_m1p2 = _shifted_overlap(model, shift_m1p2, fm_dict, vparams_dict)

    # Prefactor
    delta_F = ovrlp_shifted_p1p2 - ovrlp_shifted_p1 - ovrlp_shifted_m1p2 + ovrlp_shifted_m1

    # Hessian
    dir_product = torch.matmul(rand_dir1, rand_dir2.transpose(0, 1)) + torch.matmul(
        rand_dir2, rand_dir1.transpose(0, 1)
    )
    hess = (4 * (epsilon**2)) ** (-1) * delta_F * dir_product

    return hess
