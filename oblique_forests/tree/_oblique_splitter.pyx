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

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf

cdef inline void _init_split(ObliqueSplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.proj_vec = 0

cdef void argsort(double[:] y, int[:] idx) nogil:

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

cdef (int, int) argmin(double[:, :] A) nogil:
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

cdef void matmul(double[:, :] A, double[:, :] B, double[:, :] res) nogil:

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
 
cdef class BaseObliqueSplitter:
    """Abstract oblique splitter class.

    Splitters are called by tree builders to find the best splits on 
    both sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  double feature_combinations, object random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # SPORF parameters
        self.feature_combinations = feature_combinations
        
        # copied from original Parth's oblique split
        # self.root_impurity = self.impurity(self.y)
        # cdef double proj_dims = 1 #max(self.max_features * self.n_features, 1)
        # cdef double n_non_zeros = 1# max(self.proj_dims * self.feature_combinations, 1)
        # self.proj_dims = proj_dims
        # self.n_non_zeros = n_non_zeros

    # def __dealloc__(self):
    #     """Destructor."""

    #     free(self.samples)
    #     free(self.features)
    #     free(self.constant_features)
    #     free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X,
                   const DOUBLE_t[:, ::1] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : DOUBLE_t*
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.

        X_idx_sorted : ndarray, default=None
            The indexes of the sorted training input samples
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = y

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, ObliqueSplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

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
    cdef void sample_proj_mat(self, double[:, :] X, 
                              double[:, :] proj_mat, double[:, :] proj_X) nogil:
        pass



# cdef class DenseObliqueSplitter(BaseObliqueSplitter):
#     cdef const DTYPE_t[:, :] X

#     cdef np.ndarray X_idx_sorted
#     cdef INT32_t* X_idx_sorted_ptr
#     cdef SIZE_t X_idx_sorted_stride
#     cdef SIZE_t n_total_samples
#     cdef SIZE_t* sample_mask

#     def __cinit__(self, Criterion criterion, SIZE_t max_features,
#                   SIZE_t min_samples_leaf, double min_weight_leaf,
#                   double feature_combinations,
#                   object random_state):

#         self.X_idx_sorted_ptr = NULL
#         self.X_idx_sorted_stride = 0
#         self.sample_mask = NULL
#     cdef int init(self,
#                   object X,
#                   const DOUBLE_t[:, ::1] y,
#                   DOUBLE_t* sample_weight,
#                   np.ndarray X_idx_sorted=None) except -1:
#         """Initialize the splitter

#         Returns -1 in case of failure to allocate memory (and raise MemoryError)
#         or 0 otherwise.
#         """

#         # Call parent init
#         Splitter.init(self, X, y, sample_weight)

#         self.X = X
#         return 0

# cdef class ObliqueSplitter(DenseObliqueSplitter):
#     def __reduce__(self):
#         """Enable pickling the splitter."""
#         return (ObliqueSplitter, (self.criterion,
#                                self.max_features,
#                                self.min_samples_leaf,
#                                self.min_weight_leaf,
#                                self.feature_combinations,
#                                self.random_state), self.__getstate__())

#     cdef void sample_proj_mat(self, double[:, :] X, 
#                               double[:, :] proj_mat, double[:, :] proj_X) nogil:
#         """Get the projection matrix and it fits the transform to the samples of interest.

#         # TODO 
#         C's rand function is not thread safe, so this block is currently with GIL.
#         When merging this code with sklearn, we can use their random number generator from their utils
#         But since I don't have that here with me, I'm using C's rand function for now.

#         proj_mat & proj_X should be np.zeros()

#         """
#         pass
        # cdef UINT32_t* random_state = &self.rand_r_state

        # # TODO: ask Parth how to fix this
        # cdef int n_non_zeros = 5 # &self.n_non_zeros

        # cdef int n_samples = X.shape[0]
        # cdef int n_features = X.shape[1]
        # cdef int proj_dims = proj_X.shape[1]
        
        # # declare indexes
        # cdef int idx, feat_i, proj_i

        # # declare weight types
        # cdef int weight

        # # Draw n non zeros & put into proj_mat
        # for idx in prange(0, n_non_zeros, nogil=True):
        #     # Draw a feature at random
        #     feat_i = rand_int(0, n_features, random_state)
        #     proj_i = rand_int(0, proj_dims, random_state)

        #     # set weights to +/- 1
        #     weight = 1 if (rand_int(0, 2, random_state) == 1) else -1
        #     proj_mat[feat_i, proj_i] = weight 
        
        # matmul(X, proj_mat, proj_X)



    # X, y are X/y relevant samples. sample_inds only passed in for sorting
    # Will need to change X to not be proj_X rn
    # cpdef node_split(self, double[:, :] X, double[:] y, int[:] sample_inds):
    #     """Find the optimal split for a set of samples.
    
    #     """
    #     cdef int n_samples = X.shape[0]     # number of samples in data
    #     cdef int proj_dims = X.shape[1]     # dimensionality of the data
    #     cdef int i = 0                      # index over samples
    #     cdef int j = 0                      # index over dimensions
    #     cdef long temp_int = 0; 
    #     cdef double node_impurity = 0;      # impurity of a node

    #     # keep track of split record for current node and the best split
    #     # found among the sampled projection vectors
    #     cdef ObliqueSplitRecord best, current

    #     # instantiate the split records
    #     _init_split(&best, end)

    #     cdef int thresh_i = 0
    #     cdef int feature = 0
    #     cdef double best_gini = 0
    #     cdef double threshold = 0
    #     cdef double improvement = 0
    #     cdef double left_impurity = 0
    #     cdef double right_impurity = 0

    #     Q = np.zeros((n_samples, proj_dims), dtype=np.float64)
    #     cdef double[:, :] Q_view = Q

    #     idx = np.zeros(n_samples, dtype=np.intc)
    #     cdef int[:] idx_view = idx

    #     y_sort = np.zeros(n_samples, dtype=np.float64)
    #     cdef double[:] y_sort_view = y_sort
        
    #     feat_sort = np.zeros(n_samples, dtype=np.float64)
    #     cdef double[:] feat_sort_view = feat_sort

    #     si_return = np.zeros(n_samples, dtype=np.intc)
    #     cdef int[:] si_return_view = si_return
        

    #     # No split or invalid split --> node impurity
    #     node_impurity = self.impurity(y)
    #     Q_view[:, :] = node_impurity
        
    #     # loop over columns of the matrix (projected feature dimensions)
    #     for j in range(0, proj_dims):
    #         # get the sorted indices along the rows (sample dimension)
    #         argsort(X[:, j], idx_view)

    #         for i in range(0, n_samples):
    #             temp_int = idx_view[i]
    #             y_sort_view[i] = y[temp_int]
    #             feat_sort_view[i] = X[temp_int, j]

    #         for i in prange(1, n_samples, nogil=True):
                
    #             # Check if the split is valid!
    #             if feat_sort_view[i-1] < feat_sort_view[i]:
    #                 Q_view[i, j] = self.score(y_sort_view, i)

    #     # Identify best split
    #     (thresh_i, feature) = argmin(Q_view)
      
    #     best_gini = Q_view[thresh_i, feature]
    #     # Sort samples by split feature
    #     argsort(X[:, feature], idx_view)
    #     for i in range(0, n_samples):
    #         temp_int = idx_view[i]

    #         # Sort X so we can get threshold
    #         feat_sort_view[i] = X[temp_int, feature]
            
    #         # Sort y so we can get left_y, right_y
    #         y_sort_view[i] = y[temp_int]
            
    #         # Sort true sample inds
    #         si_return_view[i] = sample_inds[temp_int]
        
    #     # Get threshold, split samples into left and right
    #     if (thresh_i == 0):
    #         threshold = node_impurity #feat_sort_view[thresh_i]
    #     else:
    #         threshold = 0.5 * (feat_sort_view[thresh_i] + feat_sort_view[thresh_i - 1])

    #     left_idx = si_return_view[:thresh_i]
    #     right_idx = si_return_view[thresh_i:]
        
    #     # Evaluate improvement
    #     improvement = node_impurity - best_gini

    #     # Evaluate impurities for left and right children
    #     left_impurity = self.impurity(y_sort_view[:thresh_i])
    #     right_impurity = self.impurity(y_sort_view[thresh_i:])

    #     return feature, threshold, left_impurity, left_idx, right_impurity, right_idx, improvement 

    """
    Python wrappers for cdef functions.
    Only to be used for testing purposes.
    """

    def test_argsort(self, y):
        idx = np.zeros(len(y), dtype=np.intc)
        argsort(y, idx)
        return idx

    def test_argmin(self, M):
        return argmin(M)

    def test_impurity(self, y):
        return self.impurity(y)

    def test_score(self, y, t):
        return self.score(y, t)

    def test_best_split(self, X, y, idx):
        return self.best_split(X, y, idx)

    def test_matmul(self, A, B):
        res = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
        matmul(A, B, res)
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

        
