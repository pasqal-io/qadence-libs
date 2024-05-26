# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

from pytest import fixture
from qadence import QNN, BasisSet, QuantumCircuit
from qadence.constructors import feature_map, hamiltonian_factory, hea
from qadence.operations import RX, RY, Z


@fixture
def textbook_qfi_model() -> QNN:
    n_qubits, n_layers = [2, 2]
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.FOURIER)
    ansatz = hea(
        n_qubits,
        n_layers,
        param_prefix="theta",
        operations=[RX],
        periodic=True,
    )
    circuit = QuantumCircuit(n_qubits, ansatz, fm)
    obs = hamiltonian_factory(n_qubits, detuning=Z)
    return QNN(circuit, [obs])


@fixture
def basic_optim_model() -> QNN:
    n_qubits, n_layers = [2, 2]
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.FOURIER)
    ansatz = hea(
        n_qubits,
        depth=n_layers,
        param_prefix="theta",
        operations=[RX, RY],
        periodic=True,
    )
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs = hamiltonian_factory(n_qubits, detuning=Z)
    model = QNN(circuit, [obs])
    return model
