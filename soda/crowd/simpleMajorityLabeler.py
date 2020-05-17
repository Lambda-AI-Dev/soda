import numpy as np
import pandas as pd
from . import crowdLabeler
from ..utils import BipartiteGraph


def _label_count_sparse(bg: BipartiteGraph, worker_weight=None):
    """
    It is assumed that each edge in the graph is a boolean vector
    """
    if worker_weight is None:
        def agg_func(iterable):
            prob = 0
            c = 0
            for _, _, e in iterable:
                prob += e
                c += 1
            return prob / c
    else:
        def agg_func(iterable):
            prob = 0
            c = 0
            for u, _, e in iterable:
                prob += worker_weight[u] * e
                c += 1
            return prob / c
    return pd.DataFrame(bg.agg_v(agg_func)).T


class SimpleMajorityLabeler(crowdLabeler):
    def __init__(self):
        self.worker_weight = None

    def set_worker_weight(self, worker_weight):
        self.worker_weight = worker_weight

    def predict_proba_dense(self, X: np.ndarray):
        """
        values in X must be boolean (0 or 1)
        """
        return X.sum(axis=0)/X.shape[0]

    def predict_proba_nan(self, X: np.ndarray, mask: np.ndarray):
        return np.nansum(X, axis=0)/mask.sum(axis=0)

    def predict_proba_sparse(self, X: BipartiteGraph):
        return _label_count_sparse(X, worker_weight=self.worker_weight)


