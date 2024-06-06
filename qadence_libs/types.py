from __future__ import annotations

from qadence.types import StrEnum

__all__ = ["BasisSet", "ReuploadScaling", "FisherApproximation"]


class BasisSet(StrEnum):
    """Basis set for feature maps."""

    FOURIER = "Fourier"
    """Fourier basis set."""
    CHEBYSHEV = "Chebyshev"
    """Chebyshev polynomials of the first kind."""


class ReuploadScaling(StrEnum):
    """Scaling for data reuploads in feature maps."""

    CONSTANT = "Constant"
    """Constant scaling."""
    TOWER = "Tower"
    """Linearly increasing scaling."""
    EXP = "Exponential"
    """Exponentially increasing scaling."""


class FisherApproximation(StrEnum):
    """Approximation to calculate the Quantum Fisher Information (QFI)."""

    EXACT = "Exact"
    """No approximation, computes the exact QFI."""
    SPSA = "SPSA"
    """Approximation via the SPSA algorithm."""
