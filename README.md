# Qadence-Libs

**Qadence-Libs** is a Python package that provides extra functionality for Qadence.

[![Linting](https://github.com/pasqal-io/qadence-libs/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/qadence-libs/actions/workflows/lint.yml)
[![Tests](https://github.com/pasqal-io/qadence-libs/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/qadence-libs/actions/workflows/test.yml)
[![Documentation](https://github.com/pasqal-io/qadence-libs/actions/workflows/build_docs.yml/badge.svg)](https://pasqal-io.github.io/qadence-libs/latest)
[![Pypi](https://badge.fury.io/py/qadence-libs.svg)](https://pypi.org/project/qadence-libs/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation guide

[PyPI](https://pypi.org/project/qadence-libs/) and can be installed using `pip` as follows:

```bash
pip install qadence_libs
```

## Contributing

Before making a contribution, please review our [code of conduct](docs/CODE_OF_CONDUCT.md).

- **Submitting Issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/qadence-libs/issues).
- **Developing in qadence:** To learn more about how to develop within `qadence`, please refer to [contributing guidelines](docs/CONTRIBUTING.md).

### Setting up qadence in development mode

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `qadence_libs` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

**WARNING**
`hatch` will not combine nicely with other environment managers such as Conda. If you still want to use Conda,
install it from source using `pip`:

```bash
# within the Conda environment
python -m pip install -e .
```

## Citation

If you use Qadence-Libs for a publication, we kindly ask you to cite our work using the following BibTex entry:

```latex
@misc{qadence-libs2024pasqal,
  url = {https://github.com/pasqal-io/qadence-libs},
  title = {Qadence Libs: {A}n {E}xperiment runner for Qadence.},
  year = {2023}
}
```

## License
Qadence-Libs is a free and open source software package, released under the Apache License, Version 2.0.
