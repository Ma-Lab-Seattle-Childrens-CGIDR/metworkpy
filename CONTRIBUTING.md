# Contributing to METWORKpy

Thank you for you interest in contributing to METWORKpy!

## Issues

If you notice a problem with METWORKpy, or would like to request an enhancement,
you can submit an issue at the
[github issue tracker](https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/metworkpy/issues).
This can include issues with the library, but also the documentation and any
other issues that you notice. For issues with the code, please include a minimal
reproducible example in your issue to make finding the problem much easier (this
isn't needed if you just want to ask a question or make a request).

## Developement Setup

The first step is forking the repository so you can create a pull request, you
can do this on github by clicking the fork button, right above the about
section.

This project uses uv for building/packaging and so you'll need to have that
installed, see
[installation inforation](https://docs.astral.sh/uv/getting-started/installation/)
for more information. Once installed, you can run
`uv sync --group test --group dev` to install all the dev and test dependencies
into a virtual environment at .venv in the root directory. Then, you'll be able
to activate that environment (instructions vary based on shell and os, see the
[venv documentation](https://docs.python.org/3/library/venv.html) for more
information).

Additionally, pre-commit hooks are used to ensure linting/formatting
consistency. The pre-commit pypi package is a development dependency, so you
shouldn't have to worry about installing it, but you can also install it
globally using uv, e.g. `uv tool install pre-commit`. Then, once you have either
installed it manually, or activated the developement virtual environment, run
`pre-commit install` to install the hooks.

To run tests/linting/formatting locally, you can use tox. You can install it
with uv with `uv tool install tox --with tox-uv` (or whatever package manager
you would prefer). Then you can run linting with `tox -e lint`, formatting with
`tox -e format`, and running tests against individual python versions with
`tox -e <VERSION>` (like `tox -e 3.12` to run tests with python 3.12).

## Pull Requests

When submitting a pull request (especially one that changes user facing code),
please make sure that there are tests for any added functionality, and also add
a note to the
[changelog file](https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/metworkpy/blob/main/CHANGELOG.md).
All pull requests will have to pass the CI, which includes testing against
various python versions, and also linting with ruff. You can check this locally
using tox (see Developement Setup section above).
