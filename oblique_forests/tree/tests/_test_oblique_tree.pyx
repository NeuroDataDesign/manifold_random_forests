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
from cython.operator import dereference
np.import_array()

from sklearn.utils import check_random_state
from libc.stdio cimport printf

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
from oblique_forests.tree._oblique_tree cimport ObliqueTree

def test_cinit():
    n_samples = 6
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 1], (n_samples, 1))

    X = np.asfortranarray(X, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DOUBLE)

    n_outputs = 1
    n_classes = np.array([n_outputs])
    criterion = Criterion(n_outputs, n_classes)

    cdef ObliqueTree tree = ObliqueTree(n_features, n_classes, n_outputs)
    cdef SIZE_t tree_n_classes = dereference(tree.n_classes)
    
    assert tree.n_features == n_features
    assert tree.n_outputs == n_outputs
    printf("%d\n", tree_n_classes)
    # assert tree.n_classes == ...
    assert tree.max_n_classes == 1
    assert tree.max_depth == 0
    assert tree.node_count == 0
    assert tree.capacity == 0
    assert tree.value == NULL
    assert tree.nodes == NULL

    assert tree.proj_vecs == NULL


# TODO
# def test_add_node():
# Check if proj_vec is getting stored in node correctly
# Check if node is stored in tree correctly
#     return

# def test_apply_dense():
#     return

# def test_resize():
#     return

# def test_resize_c():
#     return

# def test_get_value_ndarray():
#     return

# def test_get_node_ndarray():
#     return


# def test_apply_sparse_csr():
#     return

# def test_decision_path_dense():
#     return
