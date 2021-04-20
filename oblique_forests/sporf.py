import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._forest import ForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import _joblib_parallel_args

from oblique_forests.tree.oblique_tree import ObliqueTreeClassifier


class ObliqueForestClassifier(ForestClassifier):
    """Sparse projection oblique forest classifier (SPORF).

    Parameters
    ----------
    n_estimators : int, optional
        [description], by default 100
    max_depth : [type], optional
        [description], by default None
    min_samples_split : int, optional
        [description], by default 2
    min_samples_leaf : int, optional
        [description], by default 1
    max_features : float, optional
        [description], by default 1.0
    feature_combinations : float, optional
        [description], by default 1.5
    bootstrap : bool, optional
        [description], by default True
    oob_score : bool, optional
        [description], by default False
    n_jobs : [type], optional
        [description], by default None
    random_state : [type], optional
        [description], by default None
    verbose : int, optional
        [description], by default 0
    warm_start : bool, optional
        [description], by default False
    class_weight : [type], optional
        [description], by default None
    max_samples : [type], optional
        [description], by default None
    """

    def __init__(
        self,
        n_estimators=100,
        #  criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        #  min_weight_fraction_leaf=0.,
        max_features=1.0,
        feature_combinations=1.5,
        #  max_leaf_nodes=None,
        #  min_impurity_decrease=0.,
        #  min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        #  ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ObliqueTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                #   "min_weight_fraction_leaf",
                "max_features",
                "feature_combinations",
                #   "max_leaf_nodes",
                #   "min_impurity_decrease", "min_impurity_split",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        # self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.feature_combinations = feature_combinations
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.min_impurity_split = min_impurity_split

    @property
    def feature_importances_(self):
        """
        Computes the importance of every unique feature used to make a split
        in each tree of the forest.

        Parameters
        ----------
        normalize : bool, default=True
            A boolean to indicate whether to normalize feature importances.

        Returns
        -------
        importances : array of shape [n_features]
            Array of count-based feature importances.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   **_joblib_parallel_args(prefer='threads'))(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_ if tree.tree.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    @property
    def feature_importances2_(self):
        """
        Computes the importance of every unique feature used to make a split
        in each tree of the forest.

        Parameters
        ----------
        normalize : bool, default=True
            A boolean to indicate whether to normalize feature importances.

        Returns
        -------
        importances : array of shape [n_features]
            Array of count-based feature importances.
        """
        # TODO: Parallelize this and see if there is an equivalent way to express this better
        # 1. Find all unique atoms in the forest
        # 2. Compute number of times each atom appears across all trees
        forest_projections = [node.proj_vec 
                 for tree in self.estimators_ if tree.tree.node_count > 0 
                 for node in tree.tree.nodes if node.proj_vec is not None]
        unique_projections, counts = np.unique(forest_projections, axis=0, return_counts=True)
        
        # 3. An atom assigns importance to each feature based on count of atom usage
        importances = np.zeros(self.n_features_, dtype=np.float64)
        for atom, count in zip(unique_projections, counts):
            importances[np.nonzero(atom)] += count
        
        # 4. Average across atoms
        if len(unique_projections) > 0:
            importances /= len(unique_projections)

        return importances
