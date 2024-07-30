# dicarlo-lab-to-nwb
NWB conversion scripts for DiCarlo lab data to the [Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.


## Installation

You can install the latest release of the package with [pip](https://pypi.org/project/dicarlo-lab-to-nwb/)

```
pip install dicarlo-lab-to-nwb
```
While we wait for the latest release of neo you will also require to install the latest version of neo from the github repository. You can do this by running the following command:

```bash
pip install "neo@git+https://github.com/NeuralEnsemble/python-neo@master"
```

To get the latest version of the code, you can install the package by cloning from github and installing

```bash
git clone https://github.com/catalystneuro/dicarlo-lab-to-nwb
cd dicarlo-lab-to-nwb
pip install -e .
```

If you already have the repository cloned, be sure to run the following instructions in your local copy of the github repository so you have the latest version of the code:

```bash
git pull
pip install -e . --upgrade
```

We recommend that you install the package inside a [virtual environment](https://docs.python.org/3/tutorial/venv.html). A simple way of doing this is to use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) from the `conda` package manager ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)). Detailed instructions on how to use conda environments can be found in their [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).



## Basic Usage
The basic usage of the package is documented in the [showcase_notebook.ipynb](https://github.com/catalystneuro/dicarlo-lab-to-nwb/blob/main/src/dicarlo_lab_to_nwb/conversion/showcase_notebook.ipynb)
