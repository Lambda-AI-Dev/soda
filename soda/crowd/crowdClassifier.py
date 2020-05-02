import numpy as np
import pandas as pd


def _prob_to_class(df):
    if isinstance(df, np.ndarray):
        return df.argmax(axis=1)
    elif isinstance(df, pd.DataFrame):
        return df.idxmax(axis=1)


class CrowdClassifier:
    """
    Assumptions:
    1. Binary or Multiclass classification (single task)
    2. m workers W = {W_0, W_1, ..., W_{m-1}}
    3. n objects/examples/data points to classify: X = {x_0, x_1, ..., x_{n-1}}
    4. l classes for each object: C = {C_0, C_1, ..., C_{l-1}}
        (the set of classes C is the same for all objects)
    5. X: (m, n, l) array of predicted probabilities
        or (m, n) array of predicted classes
    6. Only the predicted labels/probability is given; we don't have access to the original features
    7. Every object is classified by at least 1 worker
    8. The algorithm runs on all classified objects after all input from workers has been collected,
        no partial fit/continuous update (IMPORTANT!)

    Methods:
    - predict: always returns (n_examples, ) array of predicted classes
    - predict_proba: always returns (n_examples, n_classes) array of predicted probability vectors
    - dense: dense np.ndarray with no missing values;
        (n_workers, n_examples) array of labels or
        (n_workers, n_examples, n_classes) array of class probabilities
    - nan: dense np.ndarray with missing values;
        (n_workers, n_examples) array of labels or
        (n_workers, n_examples, n_classes) array of class probabilities (an entire row can be nan)
    - sparse: bipartite graph

    Input representations:
    X: (n_workers, n_examples) -> worker i classifies object j as class X[i, j]
    X: (n_workers, n_examples, n_classes) -> worker i believes the probability that object j belongs
        to class k is X[i, j, k]
    U, V, E: triplets of (worker, example, prediction). A prediction can be either
    """
    def predict(self, X):
        try:
            if isinstance(X, np.ndarray):
                mask = np.isfinite(X)
                if mask.all():
                    return self.predict_dense(X)
                else:
                    return self.predict_nan(X, mask)
            else:
                U, V, E = X
                return self.predict_sparse(U, V, E)
        except NotImplementedError:
            return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X, **kwargs):
        if isinstance(X, np.ndarray):
            mask = np.isfinite(X)
            if mask.all():
                return self.predict_proba_dense(X)
            else:
                return self.predict_proba_nan(X, mask)
        else:
            return self.predict_proba_sparse(X)

    def predict_dense(self, X: np.ndarray):
        return _prob_to_class(self.predict_proba_dense(X))

    def predict_proba_dense(self, X: np.ndarray):
        raise NotImplementedError

    def predict_nan(self, X: np.ndarray, mask: np.ndarray):
        return _prob_to_class(self.predict_proba_nan(X, mask))

    def predict_proba_nan(self, X: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

    def predict_sparse(self, X):
        return _prob_to_class(self.predict_proba_sparse(X))

    def predict_proba_sparse(self, X):
        raise NotImplementedError

    def fit_dense(self, X, y, sample_weight):
        raise NotImplementedError

    def fit_nan(self, X, y, sample_weight):
        raise NotImplementedError

    def fit_sparse(self, X, y, sample_weight):
        raise NotImplementedError
