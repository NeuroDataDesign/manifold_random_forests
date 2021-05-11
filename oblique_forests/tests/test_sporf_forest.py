import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
)

import pytest

from oblique_forests.sporf import ObliqueForestClassifier, PythonObliqueForestClassifier


def test_sparse_parity_py():
    clf = PythonObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/sparse_parity_train_1000.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/sparse_parity_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.8


def test_sparse_parity():

    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/sparse_parity_train_1000.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/sparse_parity_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.8


def test_orthant():

    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/orthant_train_400.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/orthant_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.95
