from __future__ import annotations

from importlib import import_module

from .constructors import *

"""Fetch the functions defined in the __all__ of each sub-module.

Import to the qadence name space. Make sure each added submodule has the respective definition:

    - `__all__ = ["function0", "function1", ...]`

Furthermore, add the submodule to the list below to automatically build
the __all__ of the qadence namespace. Make sure to keep alphabetical ordering.
"""

list_of_submodules = [
    ".constructors",
    ".qinfo_tools",
]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(import_module(submodule, package="qadence_libs"), "__all__")
    __all__ += __all_submodule__
