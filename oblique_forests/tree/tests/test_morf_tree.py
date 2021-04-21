# from the template test in sklearn:
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/tests/test_template.py

import math
import re

import numpy as np
import pytest
from sklearn import datasets, metrics
from sklearn.utils.validation import check_random_state

from oblique_forests.tree.morf_split import Conv2DSplitter
from oblique_forests.sporf import ObliqueForestClassifier as SPORF
from oblique_forests.tree.oblique_tree import ObliqueTreeClassifier as OTC
from oblique_forests.morf import Conv2DObliqueForestClassifier as MORF
from oblique_forests.tree.morf_tree import Conv2DObliqueTreeClassifier

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [0, 0, 0, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [0, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_convolutional_splitter():
    random_state = 12345
    n = 50
    height = 40
    d = 40
    X = np.ones((n, height, d))
    y = np.ones((n,))
    y[:25] = 0

    splitter = Conv2DSplitter(
        X.reshape(n, -1),
        y,
        max_features=1,
        feature_combinations=1.5,
        random_state=random_state,
        image_height=height,
        image_width=d,
        patch_height_max=2,
        patch_height_min=2,
        patch_width_max=3,
        patch_width_min=3,
    )

    splitter.sample_proj_mat(splitter.indices)


if __name__ == "__main__":

    test_convolutional_splitter()

    # from sklearn.datasets import fetch_openml
    from keras.datasets import mnist
    import time

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Get 100 samples of 3s and 5s
    num = 100
    threes = np.where(y_train == 3)[0][:num]
    fives = np.where(y_train == 5)[0][:num]
    train_idx = np.concatenate((threes, fives))

    # Subset train data
    Xtrain = X_train[train_idx]
    ytrain = y_train[train_idx]

    # Apply random shuffling
    permuted_idx = np.random.permutation(len(train_idx))
    Xtrain = Xtrain[permuted_idx]
    ytrain = ytrain[permuted_idx]

    # Subset test data
    test_idx = np.where(y_test == 3)[0]
    Xtest = X_test[test_idx]
    ytest = y_test[test_idx]

    print(f"-----{2 * num} samples")

    clf = OTC(random_state=0)
    start = time.time()
    clf.fit(Xtrain.reshape(Xtrain.shape[0], -1), ytrain)
    elapsed = time.time() - start
    print(elapsed)
    print(f"SPORF Tree: {elapsed} sec")

    clf = Conv2DObliqueTreeClassifier(image_height=28, image_width=28, random_state=0)
    start = time.time()
    clf.fit(Xtrain.reshape(Xtrain.shape[0], -1), ytrain)
    elapsed = time.time() - start
    print(f"MORF Tree: {elapsed} sec")
    
    clf = SPORF(n_estimators=100, random_state=0)
    start = time.time()
    clf.fit(Xtrain.reshape(Xtrain.shape[0], -1), ytrain)
    elapsed = time.time() - start
    print(f"SPORF: {elapsed} sec")