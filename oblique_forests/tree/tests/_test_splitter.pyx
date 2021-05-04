#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3
#cython: binding=True

import functools
import pytest
import inspect
import numpy as np
cimport numpy as np
np.import_array()

from sklearn.utils import check_random_state

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from oblique_forests.tree._criterion import Criterion
from oblique_forests.tree._oblique_splitter import ObliqueSplitter
from oblique_forests.tree._oblique_splitter cimport BaseObliqueSplitter


def test_init():
    n_samples = 6
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 1], (n_samples, 1))

    X = np.asfortranarray(X, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DOUBLE)

    n_outputs = 1
    n_classes = np.array([n_outputs])
    criterion = Criterion(n_outputs, n_classes)

    max_features = 2
    min_samples_leaf = 1
    min_weight_leaf = 0.
    feature_combinations = 1.5
    random_state = check_random_state(0)
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef DOUBLE_t* sample_weight_ptr = NULL

    assert splitter.proj_mat == NULL
    assert splitter.init(X, y, sample_weight_ptr) == 0
    assert splitter.proj_mat != NULL


def test_sample_proj_mat():
    n_samples = 6
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 1], (n_samples, 1))
    
    X = np.asfortranarray(X, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DOUBLE)

    n_outputs = 1
    n_classes = np.array([n_outputs])
    criterion = Criterion(n_outputs, n_classes)

    max_features = 2
    min_samples_leaf = 1
    min_weight_leaf = 0.
    feature_combinations = 1.5
    random_state = check_random_state(0)
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef DOUBLE_t* sample_weight_ptr = NULL
    splitter.init(X, y, sample_weight_ptr)

    # Projection matrix of splitter initializes to all zeros
    n_nonzeros = 0
    for i in range(splitter.max_features):
        for j in range(splitter.n_features):
            if splitter.proj_mat[i][j] != 0:
                n_nonzeros += 1
    assert n_nonzeros == 0

    # Sample projections in place using proj_mat pointer
    proj_mat = splitter.proj_mat
    splitter.sample_proj_mat(proj_mat)

    # Projection matrix of splitter now has at least one nonzero
    n_nonzeros = 0
    for i in range(splitter.max_features):
        for j in range(splitter.n_features):
            if splitter.proj_mat[i][j] != 0:
                n_nonzeros += 1
    assert n_nonzeros > 0

# TODO
# def test_node_reset():
#     assert False

# TODO
# def test_node_split():
#     assert False

# TODO
# def test_node_value():
#     assert False

# TODO
# def test_node_impurity():
#     assert False
