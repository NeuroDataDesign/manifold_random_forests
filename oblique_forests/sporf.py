from sklearn.ensemble._forest import ForestClassifier

from .oblique_tree import ObliqueTreeClassifier


class ObliqueForestClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        #  criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        #  min_weight_fraction_leaf=0.,
        max_features=1.0,
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
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.min_impurity_split = min_impurity_split

    
