#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: binding=True
#cython: embedsignature=True

import functools
import pytest
import inspect
import numpy as np
cimport numpy as np
np.import_array()

from oblique_forests.tree._criterion import Criterion
from oblique_forests.tree._oblique_splitter import ObliqueSplitter


def test_dummy():
    assert True


def test_split():
    n_outputs = 1
    n_classes = np.array([n_outputs])
    criterion = Criterion(n_outputs, n_classes)

    max_features = 2
    min_samples_leaf = 1,
    min_weight_leaf = 0.,
    feature_combinations = 1.5
    random_state = 0
    #splitter = ObliqueSplitter(criterion, max_features, 
    #                           min_samples_leaf, min_weight_leaf, 
    #                           feature_combinations, random_state)