import numpy as np
import numba

from soda.utils.array import array_normalize
from soda.utils.graph import BipartiteGraph
from soda.metrics.accuracyScore import accuracy_score
from soda.crowd.crowdClassifier import CrowdClassifier


@numba.jit
def _class_count_2(X, n_classes, worker_weight=None):
    prob = np.zeros((X.shape[1], n_classes))
    if worker_weight is None:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                prob[j, X[i, j]] += 1
    else:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                prob[j, X[i, j]] += worker_weight[i]
    return prob


@numba.jit
def _class_count_nan_2(X, mask, n_classes, worker_weight=None):
    prob = np.zeros((X.shape[1], n_classes))
    if worker_weight is None:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    prob[j, X[i, j]] += 1
    else:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    prob[j, X[i, j]] += worker_weight[i]
    return prob


@numba.jit
def _class_count_nan_3(X, mask, n_classes, worker_weight=None):
    prob = np.zeros((X.shape[1], n_classes))
    if worker_weight is None:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    prob[j] += X[i, j]
    else:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    prob[j] += worker_weight[i] * X[i, j]
    return prob


@numba.jit
def _class_count_sparse_1(U, V, E, n_examples, n_classes, worker_weight=None):
    """
    Each 3-tuple (u, v, e) in (U, V, E) is a valid annotation:
    u: worker index
    v: object index
    e: class index (scalar/integer); thus ndim=1
    """
    prob = np.zeros((n_examples, n_classes))
    if worker_weight is None:
        for i in range(U.shape[0]):
            prob[V[i], E[i]] += 1
    else:
        for i in range(U.shape[0]):
            prob[V[i], E[i]] += worker_weight[U[i]]
    return prob


@numba.jit
def _class_count_sparse_2(U, V, E, n_examples, n_classes, worker_weight=None):
    """
    Each 3-tuple (u, v, e) in (U, V, E) is a valid annotation:
    (u: worker index; v: object index; e: class probability vector); thus ndim = 2
    """
    prob = np.zeros((n_examples, n_classes))
    if worker_weight is None:
        for i in range(U.shape[0]):
            prob[V[i]] += E[i]
    else:
        for i in range(U.shape[0]):
            prob[V[i]] += E[i] * worker_weight[U[i]]
    return prob


class SimpleMajorityClassifier(CrowdClassifier):
    """
    tie-breaking behavior may be undefined

    weight_func determines the algorithm's behavior on how to convert an individual accuracy to worker weights
    """
    def __init__(self, n_classes, weight_func=None):
        self.n_classes = n_classes
        self.worker_weight = None
        self.weight_func = weight_func
        if weight_func is None or weight_func == "accuracy":
            self.weight_func = lambda d: d  # using accuracy as
        # other examples of weight func:
        # lambda d: d > 0.5  # only admits workers who do better than 0.5; full trust once admitted

    def predict_proba_dense(self, X: np.ndarray, **kwargs):
        """
        Args:
        X: (n_workers, n_examples) array.
            X[i, j] represents the class label given by worker i to example j (integer in [0, self.n_classes-1])
            Each worker's label can be seen as a lower-level estimator and this is a meta-model.
            Unlike stacking, the EM algorithm doesn't require k-fold training.
        or (n_workers, n_examples, n_classes) array of predicted probability

        worker_weight: (n_workers, ) array

        Returns:
        prob: (n_examples, n_classes) array of predicted probabilities
        """
        if X.ndim == 2:  # (n_workers, n_examples)
            prob = _class_count_2(X, n_classes=self.n_classes, worker_weight=self.worker_weight)
            return array_normalize(prob, axis=1)
        elif X.ndim == 3:  # (n_workers, n_examples, n_classes)
            if self.worker_weight is None:
                prob = X.sum(axis=0)
            else:
                prob = (self.worker_weight.reshape([-1, -1, 1]) * X).sum(axis=0)
            return array_normalize(prob, axis=1)
        else:
            raise ValueError(f"Cannot handle input X of shape {X.shape}")

    def predict_proba_nan(self, X: np.ndarray, mask: np.ndarray, worker_weight=None, **kwargs):
        count_func = _class_count_nan_2 if X.ndim == 2 else _class_count_nan_3
        prob = count_func(X, mask, n_classes=self.n_classes, worker_weight=worker_weight)
        return array_normalize(prob, axis=1)

    def predict_proba_sparse(self, U, V, E, n_examples, worker_weight=None, **kwargs):
        count_func = _class_count_sparse_1 if E.ndim == 1 else _class_count_sparse_2
        prob = count_func(U, V, E, n_classes=self.n_classes, worker_weight=worker_weight)
        return array_normalize(prob, axis=1)

    def fit_dense(self, X, y, sample_weight=None):
        self.worker_weight = np.array([accuracy_score(y_pred=x, y_true=y, sample_weight=sample_weight) for x in X])

    def fit_sparse(self, U, V, E, y_gold, sample_weight=None):
        """
        y_gold[v] maps from task/item index to its true label
        sample_weight[v] maps from task/item index to its weight
        """
        bg = BipartiteGraph()
        worker_accuracy = None
        if sample_weight is None:
            bg.add_edges_t(U, V, E)

            def agg_func(lst):
                true_list, pred_list = [], []
                for _, v, e in lst:
                    true_list.append(y_gold[v])
                    pred_list.append(e)
                return accuracy_score(y_true=np.array(true_list), y_pred=np.array(pred_list), normalize=True)

            worker_accuracy = bg.agg_u(agg_func)
        else:
            bg.add_edges_t(U, V, E)

            def agg_func(lst):
                weight_list, true_list, pred_list = [], [], []
                for _, v, e in lst:
                    weight_list.append(sample_weight[v])
                    true_list.append(y_gold[v])
                    pred_list.append(e)
                return accuracy_score(
                    y_true=np.array(true_list), y_pred=np.array(pred_list),
                    sample_weight=np.array(weight_list), normalize=True
                )
            worker_accuracy = bg.agg_u(agg_func)
        self.worker_weight = {u: self.weight_func(worker_accuracy[u]) for u in worker_accuracy}
