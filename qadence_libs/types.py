from __future__ import annotations

from qadence.types import StrEnum

__all__ = ["BasisSet", "ReuploadScaling"]


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
