<!-- [![Codecov](https://codecov.io/gh/adam2392/mne-hfo/branch/master/graph/badge.svg)](https://codecov.io/gh/adam2392/mne-hfo)
![.github/workflows/main.yml](https://github.com/adam2392/mne-hfo/workflows/.github/workflows/main.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/adam2392/mne-hfo.svg?style=svg)](https://circleci.com/gh/adam2392/mne-hfo)
![License](https://img.shields.io/pypi/l/mne-bids)
[![Code Maintainability](https://api.codeclimate.com/v1/badges/3afe97439ec5133ce267/maintainability)](https://codeclimate.com/github/adam2392/mne-hfo/maintainability)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) -->

Morf
====

Original code from SPORF to cythonize for the intentions of sklearn PR.

Installation
------------
Using conda instructions from sklearn:

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
    joblib threadpoolctl pytest compilers llvm-openmp

    conda activate sklearn-dev
    pipenv install --dev --skip-lock
    
    make clean

    make build-dev

Installation is RECOMMENDED via a python virtual environment, using ``pipenv``. The package is hosted on ``pypi``, which
can be installed via pip, or pipenv.

    python3.8 -m venv .venv
    pip install --upgrade pip
    pip install --upgrade pipenv

    # activate virtual environment
    pipenv shell

    # install packages using pipenv
    pipenv install --dev --skip-lock

    # if that doesn't work, just use the requirements.txt file
    pip install -r requirements.txt

then use Makefile recipe to build dev version. You'll need Cython installed.

    make build-dev


Documentation and Usage
-----------------------

The documentation can be found under the following links:

- for the [stable release](https://mne-hfo.readthedocs.io/en/stable/index.html)
- for the [latest (development) version](https://mne-hfo.readthedocs.io/en/latest/index.html)

Note: Functionality has been tested on MacOSX and Ubuntu.
