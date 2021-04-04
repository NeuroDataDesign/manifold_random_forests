from sklearn.ensemble._forest import ForestClassifier
from oblique_tree import ObliqueTreeClassifier

class ObliqueForestClassifier(ForestClassifier):

    def __init__(
            self,
            *,
            n_estimators=100,

            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0,

            max_features=1.0,
            feature_combinations=1.5,

    ):

        super(ObliqueTreeClassifier, self).__init__(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease,
                feature_combinations=feature_combinations,
                max_features=max_features,
                random_state=random_state,
        )

        
    def fit(self, X, y):

        random_state = check_random_state(self.random_state)


