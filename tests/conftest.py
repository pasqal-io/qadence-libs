# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

from pytest import fixture
from qadence import QNN, BasisSet, QuantumCircuit
from qadence.constructors import feature_map, hamiltonian_factory, hea
from qadence.operations import RX, RY, Z

N_QUBITS_OPTIM = 2
N_LAYERS_OPTIM = 2


@fixture
def basic_optim_circuit() -> QuantumCircuit:
    fm = feature_map(N_QUBITS_OPTIM, range(N_QUBITS_OPTIM), param="phi", fm_type=BasisSet.FOURIER)
    ansatz = hea(
        N_QUBITS_OPTIM, N_LAYERS_OPTIM, param_prefix="theta", operations=[RX], periodic=True
    )
    circuit = QuantumCircuit(N_QUBITS_OPTIM, ansatz, fm)
    return circuit


@fixture
def basic_optim_model() -> tuple[QuantumCircuit, QNN]:
    fm = feature_map(N_QUBITS_OPTIM, range(N_QUBITS_OPTIM), param="phi", fm_type=BasisSet.FOURIER)
    ansatz = hea(
        N_QUBITS_OPTIM,
        depth=N_LAYERS_OPTIM,
        param_prefix="theta",
        operations=[RX, RY],
        periodic=True,
    )
    circuit = QuantumCircuit(N_QUBITS_OPTIM, fm, ansatz)
    obs = hamiltonian_factory(N_QUBITS_OPTIM, detuning=Z)
    model = QNN(circuit, [obs])
    return circuit, model
