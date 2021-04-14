import numpy as np
import numpy.random as rng

from scipy.sparse import issparse
from sklearn.base import is_classifier
from sklearn.tree import _tree
from sklearn.utils import check_random_state

# from .transform import TransformationMixin
from ._split import BaseObliqueSplitter
from .oblique_tree import ObliqueSplitter, ObliqueTree, ObliqueTreeClassifier


def _check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class Conv2DSplitter(ObliqueSplitter):
    """Convolutional splitter.

    A class used to represent a 2D convolutional splitter, where splits
    are done on a convolutional patch.

    Note: The convolution function is currently just the
    summation operator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    image_height : int, optional (default=None)
        MORF required parameter. Image height of each observation.
    image_width : int, optional (default=None)
        MORF required parameter. Width of each observation.
    patch_height_max : int, optional (default=max(2, floor(sqrt(image_height))))
        MORF parameter. Maximum image patch height to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_height)))``.
    patch_height_min : int, optional (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    patch_width_max : int, optional (default=max(2, floor(sqrt(image_width))))
        MORF parameter. Maximum image patch width to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_width)))``.
    patch_width_min : int (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    discontiguous_height : bool, optional (defaul=False)
        Whether or not the rows of the patch are taken discontiguously or not.
    discontiguous_width : bool, optional (default=False)
        Whether or not the columns of the patch are taken discontiguously or not.

    Methods
    -------
    sample_proj_mat
        Will compute projection matrix, which has columns as the vectorized
        convolutional patches.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height: bool = False,
        discontiguous_width: bool = False,
        debug: bool = False,
    ):
        super(Conv2DSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )
        # set sample dimensions
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height_max = patch_height_max
        self.patch_width_max = patch_width_max
        self.patch_height_min = patch_height_min
        self.patch_width_min = patch_width_min
        self.axis_sample_dims = [
            (patch_height_min, patch_height_max),
            (patch_width_min, patch_width_max),
        ]
        self.structured_data_shape = [image_height, image_width]
        self.discontiguous_height = discontiguous_height
        self.disontiguous_width = discontiguous_width
        self.debug = debug

    def _get_rand_patch_idx(self, rand_height, rand_width):
        """Generates a random patch on the original data to consider as feature combination.

        This function assumes that data samples were vectorized. Thus contiguous convolutional
        patches are defined based on the top left corner. If the convolutional patch
        is "discontiguous", then any random point can be chosen.

        TODO:
        - refactor to optimize for discontiguous and contiguous case
        - currently pretty slow because being constructed and called many times

        Returns
        -------
        height_width_top : tuple of (height, width, topleft point)
            [description]
        """
        # XXX: results in edge effect on the RHS of the structured data...
        # compute the difference between the image dimension and current random
        # patch dimension
        delta_height = self.image_height - rand_height + 1
        delta_width = self.image_width - rand_width + 1

        # sample the top left pixel from available pixels now
        top_left_point = rng.randint(delta_width * delta_height)

        # convert the top left point to appropriate index in full image
        vectorized_start_idx = int(
            (top_left_point % delta_width)
            + (self.image_width * np.floor(top_left_point / delta_width))
        )

        # get the (x_1, x_2) coordinate in 2D array of sample
        multi_idx = self._compute_vectorized_index_in_data(vectorized_start_idx)

        if self.debug:
            print(vec_idx, multi_idx, rand_height, rand_width)

        # get random row and column indices
        if self.discontiguous_height:
            row_idx = np.random.choice(
                self.image_height, size=rand_height, replace=False
            )
        else:
            row_idx = np.arange(multi_idx[0], multi_idx[0] + rand_height)
        if self.disontiguous_width:
            col_idx = np.random.choice(self.image_width, size=rand_width, replace=False)
        else:
            col_idx = np.arange(multi_idx[1], multi_idx[1] + rand_width)

        # create index arrays in the 2D image
        structured_patch_idxs = np.ix_(
            row_idx,
            col_idx,
        )

        # get the patch vectorized indices
        patch_idxs = self._compute_index_in_vectorized_data(structured_patch_idxs)

        return patch_idxs

    def _compute_index_in_vectorized_data(self, idx):
        return np.ravel_multi_index(
            idx, dims=self.structured_data_shape, mode="raise", order="C"
        )

    def _compute_vectorized_index_in_data(self, vec_idx):
        return np.unravel_index(vec_idx, shape=self.structured_data_shape, order="C")

    def project_data(self, sample_inds):
        proj_mat = self.sample_proj_mat(sample_inds)

        # apply summation operation over the sampled patch
        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # generate random heights and widths of the patch. Note add 1 because numpy
        # needs is exclusive of the high end of interval
        rand_heights = rng.randint(
            self.patch_height_min, self.patch_height_max + 1, size=self.proj_dims
        )
        rand_widths = rng.randint(
            self.patch_width_min, self.patch_width_max + 1, size=self.proj_dims
        )

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            rand_height = rand_heights[idx]
            rand_width = rand_widths[idx]
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        return proj_mat


class GaborSplitter(ObliqueSplitter):
    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        image_height=None,
        image_width=None,
        frequency=None,
        theta=None,
        bandwidth=1,
        sigma_x=None,
        sigma_y=None,
        n_stds=3,
        offset=0,
    ):
        super(GaborSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )
        # set sample dimensions
        self.image_height = image_height
        self.image_width = image_width
        self.structured_data_shape = [image_height, image_width]

        # filter parameters
        self.frequency = frequency
        self.theta = theta
        self.bandwidth = bandwidth
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.n_stds = n_stds
        self.offset = offset

    def project_data(self, sample_inds):
        proj_mat = self.sample_proj_mat(sample_inds)

        try:
            from skimage.filters import gabor_kernel
        except Exception as e:
            raise ImportError("This function requires scikit-image.")

        frequency = rng.rand()
        theta = rng.uniform() * 2 * np.pi
        # bandwidth

        # apply summation operation over the sampled patch
        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # generate random heights and widths of the patch. Note add 1 because numpy
        # needs is exclusive of the high end of interval
        rand_heights = rng.randint(
            self.patch_height_min, self.patch_height_max + 1, size=self.proj_dims
        )
        rand_widths = rng.randint(
            self.patch_width_min, self.patch_width_max + 1, size=self.proj_dims
        )

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            rand_height = rand_heights[idx]
            rand_width = rand_widths[idx]
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        return proj_mat


class SampleGraphSplitter(ObliqueSplitter):
    """Convolutional splitter.

    A class used to represent a 2D convolutional splitter, where splits
    are done on a convolutional patch.

    Note: The convolution function is currently just the
    summation operator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    image_height : int, optional (default=None)
        MORF required parameter. Image height of each observation.
    image_width : int, optional (default=None)
        MORF required parameter. Width of each observation.
    patch_height_max : int, optional (default=max(2, floor(sqrt(image_height))))
        MORF parameter. Maximum image patch height to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_height)))``.
    patch_height_min : int, optional (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    patch_width_max : int, optional (default=max(2, floor(sqrt(image_width))))
        MORF parameter. Maximum image patch width to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_width)))``.
    patch_width_min : int (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    discontiguous_height : bool, optional (defaul=False)
        Whether or not the rows of the patch are taken discontiguously or not.
    discontiguous_width : bool, optional (default=False)
        Whether or not the columns of the patch are taken discontiguously or not.

    Methods
    -------
    sample_proj_mat
        Will compute projection matrix, which has columns as the vectorized
        convolutional patches.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        sample_strategies: list,
        sample_dims: list,
        patch_dims: list = None,
    ):
        super(SampleGraphSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )

        if axis_sample_graphs is None and axis_data_dims is None:
            raise RuntimeError(
                "Either the sample graph must be instantiated, or "
                "the data dimensionality must be specified. Both are not right now."
            )

        # error check sampling graphs
        if axis_sample_graphs is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_sample_graphs) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of sample graphs \
                ({len(axis_sample_graphs)}) must match \
                the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )
            if not all([x.ndim == 2 for x in axis_sample_graphs]):
                raise ValueError(
                    f"All axis sample graphs must be \
                                    2D matrices."
                )
            if not all([x.shape[0] == x.shape[1] for x in axis_sample_graphs]):
                raise ValueError(f"All axis sample graphs must be " "square matrices.")

            # XXX: could later generalize to remove this condition
            if not all([_check_symmetric(x) for x in axis_sample_graphs]):
                raise ValueError("All axis sample graphs must" "be symmetric.")

        # error check data dimensions
        if axis_data_dims is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_data_dims) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of data dimensions "
                    "({len(axis_data_dims)}) must match "
                    "the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )

            if X.shape[1] != np.sum(axis_data_dims):
                raise ValueError(
                    f"The specified data dimensionality "
                    "({np.sum(axis_data_dims)}) does not match the dimensionality "
                    "of the data (i.e. # columns in X: {X.shape[1]})."
                )

        # set sample dimensions
        self.structured_data_shape = sample_dims
        self.sample_dims = sample_dims
        self.sample_strategies = sample_strategies

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # generate random heights and widths of the patch. Note add 1 because numpy
        # needs is exclusive of the high end of interval
        rand_heights = rng.randint(
            self.patch_height_min, self.patch_height_max + 1, size=self.proj_dims
        )
        rand_widths = rng.randint(
            self.patch_width_min, self.patch_width_max + 1, size=self.proj_dims
        )

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            rand_height = rand_heights[idx]
            rand_width = rand_widths[idx]
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat
