from sklearn.ensemble._forest import ForestClassifier

from .tree.oblique_tree import ObliqueTreeClassifier
from .tree import DecisionTreeClassifier

class PythonObliqueForestClassifier(ForestClassifier):
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
            base_estimator=ObliqueForestClassifier(),
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
            base_estimator=DecisionTreeClassifier(),
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

class PyObliqueForestClassifier(ForestClassifier):
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
                # "random_state",
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
