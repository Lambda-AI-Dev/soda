import numpy as np


class CrowdLabeler:
    """
    Assumptions
    -----------
    1. Multilabel (binary) classification/labeling
    2. m workers
    3. n objects/instances/examples
    4. l labels to choose from (one example can have multiple labels)
    5. X: (m, n, l) array of boolean labels (workers, examples, labels)
       X: (m, n, l) array of probabilities (individual, multilabel binary/one vs all)
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
        return self.predict_proba_dense(X).astype(bool)

    def predict_proba_dense(self, X: np.ndarray):
        raise NotImplementedError

    def predict_nan(self, X: np.ndarray, mask: np.ndarray):
        return self.predict_proba_nan(X, mask).astype(bool)

    def predict_proba_nan(self, X: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

    def predict_sparse(self, X):
        return self.predict_proba_sparse(X).astype(bool)

    def predict_proba_sparse(self, X):
        raise NotImplementedError

    def fit_dense(self, X, y, sample_weight):
        raise NotImplementedError

    def fit_nan(self, X, y, sample_weight):
        raise NotImplementedError

    def fit_sparse(self, X, y, sample_weight):
        raise NotImplementedError



