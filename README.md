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
    
    # install files from Pipfile
    pip install pipenv 
    pipenv install --dev --skip-lock

    # or install via requirements.txt
    pip install -r requirements.txt

    make clean

    make build-dev

To install the necessary development packages, run:

    pip install -r test_requirements.txt

    # check code style
    make pep

then use Makefile recipe to build dev version. You'll need Cython installed.

    make build-dev

Alpha Functionality
-------------------

We can impose a Gabor or wavelet filter bank. To do so, install ``skimage`` and ``pywavelets``.

    pip install scikit-image
    pip install PyWavelets


Using with Jupyter Notebook
---------------------------

To setup an ipykernel with jupyter notebook, then do:

    python -m ipykernel install --name sklearn --user 
