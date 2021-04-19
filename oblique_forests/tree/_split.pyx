#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

cimport cython

import numpy as np

from libcpp.unordered_map cimport unordered_map
from cython.operator import dereference, postincrement

from libcpp.algorithm cimport sort as stdsort

from libcpp.vector cimport vector
from libcpp.pair cimport pair

from cython.parallel import prange

# TODO: Replace this once we parallelize the tree, since rand is not thread safe.
#from libc.stdlib import rand_int

# 0 < t < len(y)


cdef class BaseObliqueSplitter:

    cdef void argsort(self, double[:] y, int[:] idx) nogil:

        cdef int length = y.shape[0]
        cdef int i = 0
        cdef pair[double, int] p
        cdef vector[pair[double, int]] v
        
        for i in range(length):
            p.first = y[i]
            p.second = i
            v.push_back(p)

        stdsort(v.begin(), v.end())

        for i in range(length):
            idx[i] = v[i].second

    cdef (int, int) argmin(self, double[:, :] A) nogil:
        cdef int N = A.shape[0]
        cdef int M = A.shape[1]
        cdef int i = 0
        cdef int j = 0
        cdef int min_i = 0
        cdef int min_j = 0
        cdef double minimum = A[0, 0]

        for i in range(N):
            for j in range(M):

                if A[i, j] < minimum:
                    minimum = A[i, j]
                    min_i = i
                    min_j = j

        return (min_i, min_j)

    cdef double impurity(self, double[:] y) nogil:
        cdef int length = y.shape[0]
        cdef double dlength = y.shape[0]
        cdef double temp = 0
        cdef double gini = 1.0
        
        cdef unordered_map[double, double] counts
        cdef unordered_map[double, double].iterator it = counts.begin()

        if length == 0:
            return 0

        # Count all unique elements
        for i in range(0, length):
            temp = y[i]
            counts[temp] += 1

        it = counts.begin()
        while it != counts.end():
            temp = dereference(it).second
            temp = temp / dlength
            temp = temp * temp
            gini -= temp

            postincrement(it)

        return gini

    cdef double score(self, double[:] y, int t) nogil:
        cdef double length = y.shape[0]
        cdef double left_gini = 1.0
        cdef double right_gini = 1.0
        cdef double gini = 0
    
        cdef double[:] left = y[:t]
        cdef double[:] right = y[t:]

        cdef double l_length = left.shape[0]
        cdef double r_length = right.shape[0]

        left_gini = self.impurity(left)
        right_gini = self.impurity(right)

        gini = (l_length / length) * left_gini + (r_length / length) * right_gini
        return gini

    cdef void matmul(self, double[:, :] A, double[:, :] B, double[:, :] res) nogil:

        cdef int i, j, k
        cdef int m, n, p

        m = A.shape[0]
        n = A.shape[1]
        p = B.shape[1]

        for i in range(m):
            for j in range(p):

                res[i, j] = 0
                for k in range(n):
                    res[i, j] += A[i, k] * B[k, j]
    
    
    # TODO
    """
    C's rand function is not thread safe, so this block is currently with GIL.
    When merging this code with sklearn, we can use their random number generator from their utils
    But since I don't have that here with me, I'm using C's rand function for now.
    """
    cdef void sample_proj_mat(self, double[:, :] X, double[:, :] proj_mat, double[:, :] proj_X):
        
        # Sample the projection matrix
        


        pass


    # X, y are X/y relevant samples. sample_inds only passed in for sorting
    # Will need to change X to not be proj_X rn
    cpdef best_split(self, double[:, :] X, double[:] y, int[:] sample_inds):

        cdef int n_samples = X.shape[0]
        cdef int proj_dims = X.shape[1]
        cdef int i = 0
        cdef int j = 0
        cdef long temp_int = 0;
        cdef double node_impurity = 0;

        cdef int thresh_i = 0
        cdef int feature = 0
        cdef double best_gini = 0
        cdef double threshold = 0
        cdef double improvement = 0
        cdef double left_impurity = 0
        cdef double right_impurity = 0

        Q = np.zeros((n_samples, proj_dims), dtype=np.float64)
        cdef double[:, :] Q_view = Q

        idx = np.zeros(n_samples, dtype=np.intc)
        cdef int[:] idx_view = idx

        y_sort = np.zeros(n_samples, dtype=np.float64)
        cdef double[:] y_sort_view = y_sort
        
        feat_sort = np.zeros(n_samples, dtype=np.float64)
        cdef double[:] feat_sort_view = feat_sort

        si_return = np.zeros(n_samples, dtype=np.intc)
        cdef int[:] si_return_view = si_return
        
        # No split or invalid split --> node impurity
        node_impurity = self.impurity(y)
        Q_view[:, :] = node_impurity
        
        # loop over columns of the matrix (projected feature dimensions)
        for j in range(0, proj_dims):
            # get the sorted indices along the rows (sample dimension)
            self.argsort(X[:, j], idx_view)

            for i in range(0, n_samples):
                temp_int = idx_view[i]
                y_sort_view[i] = y[temp_int]
                feat_sort_view[i] = X[temp_int, j]

            for i in prange(1, n_samples, nogil=True):
                
                # Check if the split is valid!
                if feat_sort_view[i-1] < feat_sort_view[i]:
                    Q_view[i, j] = self.score(y_sort_view, i)

        # Identify best split
        (thresh_i, feature) = self.argmin(Q_view)
      
        best_gini = Q_view[thresh_i, feature]
        # Sort samples by split feature
        self.argsort(X[:, feature], idx_view)
        for i in range(0, n_samples):
            temp_int = idx_view[i]

            # Sort X so we can get threshold
            feat_sort_view[i] = X[temp_int, feature]
            
            # Sort y so we can get left_y, right_y
            y_sort_view[i] = y[temp_int]
            
            # Sort true sample inds
            si_return_view[i] = sample_inds[temp_int]
        
        # Get threshold, split samples into left and right
        if (thresh_i == 0):
            threshold = node_impurity #feat_sort_view[thresh_i]
        else:
            threshold = 0.5 * (feat_sort_view[thresh_i] + feat_sort_view[thresh_i - 1])

        left_idx = si_return_view[:thresh_i]
        right_idx = si_return_view[thresh_i:]
        
        # Evaluate improvement
        improvement = node_impurity - best_gini

        # Evaluate impurities for left and right children
        left_impurity = self.impurity(y_sort_view[:thresh_i])
        right_impurity = self.impurity(y_sort_view[thresh_i:])

        return feature, threshold, left_impurity, left_idx, right_impurity, right_idx, improvement 

    """
    Python wrappers for cdef functions.
    Only to be used for testing purposes.
    """

    def test_argsort(self, y):
        idx = np.zeros(len(y), dtype=np.intc)
        self.argsort(y, idx)
        return idx

    def test_argmin(self, M):
        return self.argmin(M)

    def test_impurity(self, y):
        return self.impurity(y)

    def test_score(self, y, t):
        return self.score(y, t)

    def test_best_split(self, X, y, idx):
        return self.best_split(X, y, idx)

    def test_matmul(self, A, B):
        res = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
        self.matmul(A, B, res)
        return res

    def test(self):

        # Test score
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        s = [self.score(y, i) for i in range(10)]
        print(s)

        # Test splitter
        # This one worked
        X = np.array([[0, 0, 0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1]], dtype=np.float64)
        y = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        si = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.intc)

        (f, t, li, lidx, ri, ridx, imp) = self.best_split(X, y, si)
        print(f, t)

        
