from __future__ import annotations

from .qfi import get_quantum_fisher, get_quantum_fisher_spsa
from .qng import QuantumNaturalGradient, QuantumNaturalGradientSPSA

# Modules to be automatically added to the qadence namespace
__all__ = [
    "QuantumNaturalGradient",
    "QuantumNaturalGradientSPSA",
    "get_quantum_fisher",
    "get_quantum_fisher_spsa",
]
