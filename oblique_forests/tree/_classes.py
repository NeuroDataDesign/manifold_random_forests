"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

import numbers
import warnings
import copy
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.tree._classes import BaseDecisionTree

from ._criterion import Criterion
from ._oblique_splitter import BaseObliqueSplitter
from ._oblique_tree import DepthFirstTreeBuilder
from ._oblique_tree import ObliqueTree
from . import _oblique_tree, _oblique_splitter, _criterion, _splitter#, _tree

__all__ = [
    "ObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor",
    #    "ExtraTreeClassifier",
    #    "ExtraTreeRegressor"
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _oblique_tree.DTYPE
DOUBLE = _oblique_tree.DOUBLE

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
# TODO: Remove "mse" in version 1.2.
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "mse": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "mae": _criterion.MAE,
    "poisson": _criterion.Poisson
}

DENSE_SPLITTERS = {
    "best": _oblique_splitter.ObliqueSplitter,
    # "random": _oblique_splitter.RandomSplitter
}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter
}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseObliqueDecisionTree(BaseDecisionTree):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        min_impurity_split,
        feature_combinations=None,
        class_weight=None,
        ccp_alpha=0.0
    ):
        super(BaseObliqueDecisionTree, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )

        # SPORF params
        self.feature_combinations = feature_combinations


    # refactor the definition of the tree into a private method
    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
    ):

        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based " "sparse matrices"
                    )

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match "
                "number of samples=%d" % (len(y), n_samples)
            )
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError(
                "max_leaf_nodes must be integral number but was " "%r" % max_leaf_nodes
            )
        if -1 < max_leaf_nodes < 2:
            raise ValueError(
                ("max_leaf_nodes {0} must be either None " "or larger than 1").format(
                    max_leaf_nodes
                )
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. Its default "
                "value has changed from 1e-7 to 0 in version 0.23, and it "
                "will be removed in 1.0 (renaming of 0.25). Use the "
                "min_impurity_decrease parameter instead.",
                FutureWarning,
            )

            if min_impurity_split < 0.0:
                raise ValueError(
                    "min_impurity_split must be greater than " "or equal to 0"
                )
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be greater than " "or equal to 0"
            )

        # TODO: Remove in 1.1
        if X_idx_sorted != "deprecated":
            warnings.warn(
                "The parameter 'X_idx_sorted' is deprecated and has no "
                "effect. It will be removed in 1.1 (renaming of 0.26). You "
                "can suppress this warning by not passing any value to the "
                "'X_idx_sorted' parameter.",
                FutureWarning,
            )

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
            # TODO: Remove in v1.2
            if self.criterion == "mse":
                warnings.warn(
                    "Criterion 'mse' was deprecated in v1.0 and will be "
                    "removed in version 1.2. Use `criterion='squared_error'` "
                    "which is equivalent.",
                    FutureWarning,
                )
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, BaseObliqueSplitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                self.feature_combinations,
                random_state,
            )

        if is_classifier(self):
            self.tree_ = ObliqueTree(self.n_features_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = ObliqueTree(
                self.n_features_,
                # TODO: tree should't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
                min_impurity_split,
            )
        # else:
        # builder = BestFirstTreeBuilder(splitter, min_samples_split,
        #                                min_samples_leaf,
        #                                min_weight_leaf,
        #                                max_depth,
        #                                max_leaf_nodes,
        #                                self.min_impurity_decrease,
        #                                min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self


# =============================================================================
# Public estimators
# =============================================================================


class ObliqueDecisionTreeClassifier(ClassifierMixin, BaseObliqueDecisionTree):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=0
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : ObliqueTree instance
        The underlying ObliqueTree object. Please refer to
        ``help(sklearn.tree._oblique_tree.ObliqueTree)`` for attributes of ObliqueTree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeRegressor : A decision tree regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        feature_combinations=1.5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        ccp_alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            feature_combinations=feature_combinations,
            ccp_alpha=ccp_alpha,
        )

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
    ):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : deprecated, default="deprecated"
            This parameter is deprecated and has no effect.
            It will be removed in 1.1 (renaming of 0.26).

            .. deprecated :: 0.24

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """

        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
        )
        return self

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, : self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class ObliqueDecisionTreeRegressor(RegressorMixin, BaseObliqueDecisionTree):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "mse", "friedman_mse", "mae", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node, and "poisson" which uses
        reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=0
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : ObliqueTree instance
        The underlying ObliqueTree object. Please refer to
        ``help(sklearn.tree._tree.ObliqueTree)`` for attributes of ObliqueTree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        feature_combinations=1.5,
        ccp_alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            feature_combinations=1.5,
            ccp_alpha=ccp_alpha,
        )

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
    ):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : deprecated, default="deprecated"
            This parameter is deprecated and has no effect.
            It will be removed in 1.1 (renaming of 0.26).

            .. deprecated :: 0.24

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """

        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
        )
        return self

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )

        self.tree_.compute_partial_dependence(
            grid, target_features, averaged_predictions
        )
        return averaged_predictions
