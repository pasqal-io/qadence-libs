# Template Python project

This is a template Python project which can be used to bootstrap a new library in the Pasqal quantum software codebase.

## Development tools

The library uses the following tools:

* [hatch](https://hatch.pypa.io/latest/) for managing virtual environment and dependencies
* [pytest](https://docs.pytest.org/en/7.2.x/contents.html) for building the unit tests suite
* [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/) and [flake8](https://flake8.pycqa.org/en/latest/) for code formatting and linting
* [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
* [pre-commit](https://pre-commit.com/) for applying linting and formatting automatically before committing new code

We recommend to use [`pyenv`](https://github.com/pyenv/pyenv) for managing
python versions for managing python versions both globally and locally:
```bash
# System-wide install of a python version.
pyenv install 3.10

# Use 3.10 everywhere.
pyenv global 3.10

# Or locally in the current directory.
pyenv local 3.10
```


## Install from registry

Before you can install the library from the private Pasqal PyPi, make sure to ask for `PYPI_USERNAME` and `PYPI_PASSWORD` on the relevant Slack channel.
You can then set the credentials as environment variables via:

```bash
export PYPI_USERNAME=MYUSERNAME
export PYPI_PASSWORD=THEPASSWORD
```

You are then able to install the latest version of `template-python-project` from the Pasqal private PyPi.


## Install from source

All Pasqal quantum libraries require Python >=3.8. For development, the preferred method to install this package is
to use `hatch`. You can install from source by cloning this repository and run:

```bash
python -m pip install hatch
python -m hatch -v shell

# execute any script using the library
python my_script.py
```

Alternatively, you can also:

* install with `pip` in development mode by simply running `pip install -e .`. Notice that in this way
  you will install all the dependencies, including extras.
* install it with `conda` by simply using `pip` inside the Conda environment.


## Develop

When developing the package, the recommended way is to create a virtual environment with `hatch` as shown above:

```bash
python -m pip install hatch
python -m hatch -v shell
```

When inside the shell with development dependencies, install first the pre-commit hook:
```
pre-commit install
```

In this way, you will get automatic linting and formatting every time you commit new code. Do not
forget to run the unit test suite by simply running the `pytest` command.

If you do not want to get into the Hatch shell, you can alternatively do the following:

```bash
python -m pip install hatch
python -m hatch -v shell

# install the pre-commit
python -m hatch run pre-commit install

# commit some code
python -m hatch run git commit -m "My awesome commit"

# run the unit tests suite
python -m hatch run pytest

```

## Document

You can improve the documentation of the package by editing this file for the landing page or adding new
markdown or Jupyter notebooks to the `docs/` folder in the root of the project. In order to modify the
table of contents, edit the `mkdocs.yml` file in the root of the project.

In order to build and serve the documentation locally, you can use `hatch` with the right environment:

```bash
python -m hatch -v run docs:build
python -m hatch -v run docs:serve
```

If you don't want to use `hatch`, just check into your favorite virtual environment and
execute the following commands:

```bash
python -m pip install -r docs/requirements.txt
mkdocs build
mkdocs serve
```
