from __future__ import annotations

from .qfi import get_quantum_fisher, get_quantum_fisher_spsa
from .qng import QNG, QNG_SPSA


# Modules to be automatically added to the qadence namespace
__all__ = [
    "QNG",
    "QNG_SPSA",
    "get_quantum_fisher",
    "get_quantum_fisher_spsa",
]
