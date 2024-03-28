from __future__ import annotations

import pytest
import torch
from torch import Size, allclose

from qadence import QNN, Overlap, QuantumCircuit, BasisSet, hamiltonian_factory
from qadence.constructors import hea, feature_map
from qadence.operations import Z, RX, RY

from qadence_libs.qinfo_tools.spsa import _shifted_overlap


def create_hea_model(n_qubits, layers):
    fm = feature_map(n_qubits, range(n_qubits), param="phi", fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=layers, param_prefix="theta", operations=[RX, RY], periodic=True)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    return circuit


# Create dummy quantum circuit for tests
TINY_CIRCUIT = create_hea_model(2, 2)
VPARAMS_VALS = [torch.Tensor([1]) for vparam in TINY_CIRCUIT.parameters() if vparam.trainable]
FM_DICT = {"phi": torch.Tensor([0])}


# Check that _shifted_overlap is 1 for shift 0
@pytest.mark.parametrize("shift", [0, 0.1])
def test_shifted_overlap(shift):
    ovrlp_model = Overlap(TINY_CIRCUIT, TINY_CIRCUIT)

    vparams_dict = {k: v for (k, v) in ovrlp_model._params.items() if v.requires_grad}
    vparams_tensors = torch.Tensor(VPARAMS_VALS).reshape((ovrlp_model.num_vparams, 1))

    ovrlp = _shifted_overlap(
        ovrlp_model,
        shift,
        FM_DICT,
        vparams_dict,
        vparams_tensors,
    )
    if shift == 0:
        print(ovrlp)
        assert torch.isclose(ovrlp, 1)
    if shift == 0.1:
        assert not torch.isclose(ovrlp, 1)


test_shifted_overlap(0)
test_shifted_overlap(0.1)
