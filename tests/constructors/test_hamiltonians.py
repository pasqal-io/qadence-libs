from __future__ import annotations

from typing import Callable

import networkx as nx
import pytest
from qadence.blocks import AbstractBlock, block_is_qubit_hamiltonian
from qadence.constructors import hamiltonian_factory
from qadence.operations import N, X, Y, Z
from qadence.register import Register
from qadence.types import Interaction
from torch import rand


def custom_interaction(i: int, j: int) -> AbstractBlock:
    return X(i) @ X(j) + Y(i) @ Y(j)


@pytest.mark.parametrize(
    "interaction",
    [None, Interaction.ZZ, Interaction.NN, Interaction.XY, Interaction.XYZ, custom_interaction],
)
@pytest.mark.parametrize("detuning", [None, X, Y, Z, N])
@pytest.mark.parametrize("strength_type", ["none", "parameter", "numeric", "random"])
def test_hamiltonian_factory_creation(
    interaction: Interaction | Callable | None,
    detuning: type[N] | type[X] | type[Z] | type[Y] | None,
    strength_type: str,
) -> None:
    n_qubits = 5

    if (interaction is None) and (detuning is None):
        pass
    else:
        detuning_strength = None
        interaction_strength = None
        random_strength = False
        if strength_type == "parameter":
            detuning_strength = "x"
            interaction_strength = "y"
        elif strength_type == "numeric":
            detuning_strength = rand(n_qubits)
            interaction_strength = rand(int(0.5 * n_qubits * (n_qubits - 1)))
        elif strength_type == "random":
            random_strength = True

        hamilt = hamiltonian_factory(
            n_qubits,
            interaction=interaction,
            detuning=detuning,
            detuning_strength=detuning_strength,
            interaction_strength=interaction_strength,
            random_strength=random_strength,
        )

        assert block_is_qubit_hamiltonian(hamilt)


@pytest.mark.parametrize(
    "register",
    [
        "graph",
        Register(4),
        Register.line(4),
        Register.circle(8),
        Register.square(4),
        Register.rectangular_lattice(2, 3),
        Register.triangular_lattice(1, 3),
        Register.honeycomb_lattice(1, 3),
        Register.from_coordinates([(0, 1), (0, 2), (0, 3), (1, 3)]),
    ],
)
@pytest.mark.parametrize("interaction", [Interaction.NN, Interaction.XY, custom_interaction])
@pytest.mark.parametrize("detuning", [Y, Z])
def test_hamiltonian_factory_register(
    register: Register | str,
    interaction: Interaction | Callable | None,
    detuning: type[N] | type[X] | type[Z] | type[Y] | None,
) -> None:
    if register == "graph":
        graph = nx.Graph()
        graph.add_edge(0, 1)
        register = Register(graph)

    hamilt = hamiltonian_factory(
        register,  # type: ignore [arg-type]
        interaction=interaction,
        detuning=detuning,
        random_strength=True,
    )

    assert block_is_qubit_hamiltonian(hamilt)
