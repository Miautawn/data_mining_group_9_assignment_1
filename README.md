# DMT Assignment 1 (Group 9)

## Setup
This codebase uses the following tools:
- [Poetry](https://python-poetry.org/): for python project and dependency management
- [Pyenv](https://github.com/pyenv/pyenv): for python version management

Please follow the links to set the tools locally.
After you're done you can continue with the repo setup:
```bash
pyenv install                   # intsalls the python version needed for the codebase
poetry intall                   # installs the needed python dependencies
poetry run pre-commit install   # installs the pre-commit hooks
```

Whenever you wish to execute a python script from the terminal, you can do it by running it through poetry:
```bash
poetry run python some_file.py
```
