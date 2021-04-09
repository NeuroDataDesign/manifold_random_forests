import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._forest import ForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import _joblib_parallel_args

from .tree.morf_tree import Conv2DObliqueTreeClassifier


class Conv2DObliqueForestClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        #  criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        #  min_weight_fraction_leaf=0.,
        max_features="auto",
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
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height=False,
        discontiguous_width=False,
    ):
        super().__init__(
            base_estimator=Conv2DObliqueTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                #   "min_weight_fraction_leaf",
                "max_features",
                #   "max_leaf_nodes",
                #   "min_impurity_decrease", "min_impurity_split",
                "random_state",
                "image_height",
                "image_width",
                "patch_height_max",
                "patch_height_min",
                "patch_width_max",
                "patch_width_max",
                "discontiguous_height",
                "discontiguous_width",
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
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.min_impurity_split = min_impurity_split

        # s-rerf params
        self.discontiguous_height = discontiguous_height
        self.discontiguous_width = discontiguous_width
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height_max = patch_height_max
        self.patch_height_min = patch_height_min
        self.patch_width_max = patch_width_max
        self.patch_width_min = patch_width_min

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