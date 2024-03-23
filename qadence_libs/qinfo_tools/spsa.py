import numpy as np
import torch
from torch import Tensor

from qadence import Overlap


def _create_random_direction(size: int):
    """Creates a torch Tensor with elements randomly drawn from {-1,+1}

    Args:
        size (int): Size of the vector

    Returns:
        torch.Tensor
    """
    return torch.Tensor(np.random.choice([-1, 1], size=(size, 1)))


def _shifted_ket_vparam_dict(shift, vparams_dict, vparams_tensors):
    return dict(zip(vparams_dict.keys(), vparams_tensors + shift))


def _shifted_overlap(model: Overlap, shifted_vparams_dict: dict, fm_dict: dict, vparams_dict: dict):
    ovrlp_shifted = model(
        bra_param_values=fm_dict | vparams_dict,
        ket_param_values=fm_dict | shifted_vparams_dict,
    )
    return ovrlp_shifted


def spsa_gradient(
    model: Overlap,
    epsilon: float,
    fm_dict: dict | None,
    vparams_values=tuple | list | Tensor | None,
):

    vparams_dict = {k: v for (k, v) in model._params.items() if v.requires_grad}
    vparams_tensors = torch.Tensor(vparams_values).reshape((model.num_vparams, 1))

    # Create random direction
    random_direction = _create_random_direction()

    # Shift ket variational parameters
    shift_plus = epsilon * random_direction
    vparams_plus = _shifted_ket_vparam_dict(shift_plus, vparams_dict, vparams_tensors)
    shift_minus = -shift_plus
    vparams_minus = _shifted_ket_vparam_dict(shift_minus, vparams_dict, vparams_tensors)

    # Overlaps with the shifted parameters
    ovrlp_shifted_plus = _shifted_overlap(model, vparams_plus, fm_dict, vparams_dict)
    ovrlp_shifted_minus = _shifted_overlap(model, vparams_minus, fm_dict, vparams_dict)

    return (ovrlp_shifted_plus - ovrlp_shifted_minus) / (2 * epsilon)


def spsa_2gradient(
    model: Overlap,
    epsilon: float,
    fm_dict: dict | None,
    vparams_values=tuple | list | Tensor | None,
):
    """_summary_

    Args:
        model (Overlap): _description_
        epsilon (float): _description_
        fm_dict (dict | None): _description_
        vparams_values (_type_, optional): _description_. Defaults to tuple | list | Tensor | None.

    Returns:
        _type_: _description_
    """

    vparams_dict = {k: v for (k, v) in model._params.items() if v.requires_grad}
    vparams_tensors = torch.Tensor(vparams_values).reshape((model.num_vparams, 1))

    # Create random directions
    rand_dir1 = _create_random_direction(size=model.num_vparams)
    rand_dir2 = _create_random_direction(size=model.num_vparams)

    # Shift ket variational parameters
    shift_p1 = epsilon * rand_dir1
    vparams_p1 = _shifted_ket_vparam_dict(shift_p1, vparams_dict, vparams_tensors)

    shift_p1p2 = epsilon * (rand_dir1 + rand_dir2)
    vparams_p1p2 = _shifted_ket_vparam_dict(shift_p1p2, vparams_dict, vparams_tensors)

    shift_m1 = -epsilon * rand_dir1
    vparams_m1 = _shifted_ket_vparam_dict(shift_m1, vparams_dict, vparams_tensors)

    shift_m1p2 = epsilon * (-rand_dir1 + rand_dir2)
    vparams_m1p2 = _shifted_ket_vparam_dict(shift_m1p2, vparams_dict, vparams_tensors)

    # Overlaps with the shifted parameters
    ovrlp_shifted_p1 = _shifted_overlap(model, vparams_p1, fm_dict, vparams_dict)
    ovrlp_shifted_p1p2 = _shifted_overlap(model, vparams_p1p2, fm_dict, vparams_dict)
    ovrlp_shifted_m1 = _shifted_overlap(model, vparams_m1, fm_dict, vparams_dict)
    ovrlp_shifted_m1p2 = _shifted_overlap(model, vparams_m1p2, fm_dict, vparams_dict)

    # Prefactor
    delta_F = ovrlp_shifted_p1p2 - ovrlp_shifted_p1 - ovrlp_shifted_m1p2 + ovrlp_shifted_m1

    # Hessian
    dir_product = torch.matmul(rand_dir1, rand_dir2.transpose(0, 1)) + torch.matmul(
        rand_dir2, rand_dir1.transpose(0, 1)
    )
    hess = (1 / 4) * (delta_F / (epsilon**2)) * dir_product

    return hess
