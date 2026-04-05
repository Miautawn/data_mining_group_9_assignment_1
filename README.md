# DMT Assignment 1 (Group 9)

## Setup
This codebase uses the following tools:
- [Poetry](https://python-poetry.org/): for python project and dependency management
- [Pyenv](https://github.com/pyenv/pyenv): for python version management
- [DVC](https://doc.dvc.org/start): for larger data artifact management

> [!NOTE]
> I recommend using [pipx](https://pipx.pypa.io/latest/how-to/install-pipx/) to install/manage poetry and dvc

Please follow the links to set the tools locally.
After you're done you can continue with the repo setup:
```bash
pyenv install                   # intsalls the python version needed for the codebase
poetry intall                   # installs the needed python dependencies
poetry run pre-commit install   # installs the pre-commit hooks
$( : "optionally" ) dvc pull    # pulls all the data artifacts (read `Using DVC` section)
```

Whenever you wish to execute a python script from the terminal, you can do it by running it through poetry:
```bash
poetry run python some_file.py
```

If you're working in jupyter notebooks from VSCode, you can use the project's virtual environment's pyhon interpreter (with all the dependencies installed) by:
1. Creating a `.ipynb` file
2. Clicking "select kernel" in the top right
3. Clicking "python environments" and  selecting the local `.venv` one

![alt text](./images/venv_selection.png)

> [!NOTE]
> You may need to install the `Jupyter` [extention](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in the VS Code

### Using DVC
While other tools might be simple to understand and operate, DVC usually requires some time getting used to, both conceptually and operationally. To that end, here's a rappid fire Q/A:

Q: **what does DVC even do?**
A: Primaraly it's for tracking and version controlling larger data artifacts (e.g. datasets, model weights)

Q: **Why can't I just commit data arficats to git like code?**
A: Fundamentally, git works on file deltas (i.e. changes). It tracks what has been changed via commit history all the way back to the original file state. Data artifacts, unlike code, are usually not changed in increments - defeating the purpose of diff tracking. Another downsite is that your repository will become sluggish as everyone would need to download these artifacts even thought they wouldn't use them. Instead, DVC replaces those artifacts with placeholders and keeps the actual artifacts in some remote storage (e.g. s3, gcs), downloading them when explicitly instructed.

Q: **What remote storage does this project use?**
A: We make use of the generosity of [DagsHub](https://dagshub.com/dashboard). They gives us a free DVC remote with very little hassle (unlike smth like google drive).

Q: **How do I work with DVC?**
A: Please read a the official [quickstart](https://doc.dvc.org/start). But fundamentally you only need the following:

1. `dvc add x` to replace `x` data artifacts with a placeholder and start tracking it using dvc. You can then commit that placeholder using git.
2. `dvc push` to push the actual data artifact to remote storage. For contributors, please see the following guide how to authorise yourself to the [DagsHub](https://dagshub.com/docs/feature_guide/dagshub_storage/#working-with-the-dvc-remote).

> [!WARNING]
> The GitHub repo has been already linked to DagsHub [here](https://dagshub.com/Miautawn/data_mining_group_9_assignment_1).
> Please request to be added as a contributor if you need to push data artifacts.

3. `dvc pull` to download the data artifacts based on your current commit (anyone can do it)
