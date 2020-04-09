"""
Crowdsourcing quality assurance via EM algorithm
Assumptions:
1. n objects to classify: O = {O_1, O_2, ..., O_n}
2. l classes for each object: C = {C_1, C_2, ..., C_l} (the set of classes C is the same for all objects)
3. m workers: W = {W_1, W_2, ..., W_m} (every worker has classified every object)
n=2

Input:
A: (m, n) matrix; A[i][j] is the class given by worker i to object j;
it must be an integer from 0 to l-1
l: the number of classes

Output:
y: (n, ) vector, true labels after convergence
s: (m, l, l) array, confusion matrix for each worker; truth * predicted
p: (l, ) vector, class priors
"""

import numpy as np
from scipy import stats
from scipy import sparse


# import sys
# sys.path.append('.')  # solve relative import problem in Python

from soda.utils.array import array_normalize
from soda.utils.graph import BipartiteGraph
from soda.metrics.confusionMatrix import confusion_matrix
from soda.crowd.crowdClassifier import CrowdClassifier



# class NaiveEMClassifier(CrowdClassifier):
#     def __init__(self, l, tol, max_iter):
#         self.n_classes = l
#         self.tol = tol
#         self.max_iter = max_iter
#         self.worker_confusion_matrices = None
#
#     def predict_proba(self, X, X_gold=None, y_gold=None):
#         """
#         X_gold.shape[0] == X.shape[0]  # the same pool of workers
#         """
#         def _step_maximize(s):  # given confusion matrices, find the label probabilities
#             return sum(s_mat[:, x] for x, s_mat in zip(X, s)).T  # x is a row of labels given by a single worker
#
#         def _step_expect(A, b):  # given true labels and workers' predictions, find confusion matrices
#             return np.array([confusion_matrix(b, a, self.n_classes, normalize=True, normalization_axis=0) for a in A])
#
#         if X_gold is None or y_gold is None:  # no gold standard; initialize true labels by majority voting
#             y_old = stats.mode(X, axis=0, nan_policy="omit").mode.flatten()
#         else:  # initialize true labels by weighted votes (from gold labels)
#             # TODO: should the gold data matter after initialization?
#             y_old = _step_maximize(_step_expect(X_gold, y_gold)).argmax(axis=0)
#         for _ in range(self.max_iter):
#             s = _step_expect(X, y_old)
#             y_new_prob = _step_maximize(s)
#             y_new = y_new_prob.argmax(axis=1)
#             if (y_old != y_new).sum() <= y_old.shape[0] * self.tol: # converges
#                 self.worker_confusion_matrices = s
#                 return array_normalize(y_new_prob, axis=1)
#             else:
#                 y_old = y_new
#         s = _step_expect(X, y_old)
#         self.worker_confusion_matrices = s
#         return array_normalize(_step_maximize(s), axis=1)
#
#     def get_worker_confuion_matrice(self):
#         return self.worker_confusion_matrices


class EMClassifier(CrowdClassifier):
    """
    This classifier also deals with missing values and sparse matrices
    """
    def __init__(self, n_classes, tol, max_iter, normalize="true"):
        self.n_classes = n_classes
        self.tol = tol
        self.max_iter = max_iter
        self.worker_confusion_matrices = None
        # the default normalization direction should be truth
        self.normalize = normalize

    @staticmethod
    def _sparse_to_graph(X):
        """
        Args:
            X: (n_workers, n_examples) sparse array

        Returns:

        """
        if not sparse.issparse(X):
            raise ValueError("Input matrix must be sparse")
        I, J, V = sparse.find(X)
        return EMClassifier._value_tuple_to_graph(I, J, V, X.shape[0], X.shape[1])

    @staticmethod
    def _value_tuple_to_graph(I, J, V, n_workers, n_examples):
        """
        Convert a tuple of sparse labeling information into a graph
        Parameters
        ----------
        I : worker indices (will be converted to nonnegative, starting from 0 to n_workers-1, inclusive)
        J : example indices (will be converted to negative, starting from -1 to -n_examples, inclusive)
        V : value or probability vector given worker index (i) and example index

        Returns
        -------

        """
        G = BipartiteGraph()
        # use negative index for examples
        G.add_edges_from(zip(I, J - n_examples, V))
        return G

    @staticmethod
    def _step_maximize_dense(X, S):
        """
        Given confusion matrices, find the label probabilities (unnormalized)

        Args:
            X: (n_workers, n_examples) array of predicted labels by each worker
                or (n_workers, n_examples, n_classes) array of predicted probabilities by each worker
            S: (n_workers, n_classes, n_classes) array of confusion matrices

        Returns:
            (n_examples, n_classes)
        """
        if X.ndim == 2:
            # x is a row of labels given by a single worker
            return sum(s[:, x] for x, s in zip(X, S)).T
        elif X.ndim == 3:
            # x is a (n_examples, n_classes) array of predicted probabilities by a single worker
            # may be more intuitive to think of (s dot x.T).T
            return sum(np.dot(x, s.T) for x, s in zip(X, S))
        else:
            raise ValueError(f"Cannot handle input worker label array of shape {X.shape}")

    @staticmethod
    def _step_maximize_nan(X, S, notnull):
        """
        Because the probability is not normalized, the differences of sums due to NaNs are kept
        Args:
            X: (n_workers, n_examples) or (n_workers, n_examples, n_classes)
            S: (n_workers, n_classes, n_classes)
            notnull: (n_workers, n_examples)
        Returns:
        """
        n_workers, n_examples = X.shape[0], X.shape[1]
        ret = np.zeros([n_workers, n_examples])
        if X.ndim == 2:
            for x, s, mask in zip(X, S, notnull):
                ret[:, mask] += s[:, x[mask]]
            return ret.T
        elif X.ndim == 3:
            for x, s, mask in zip(X, S, notnull):
                ret[:, mask] += np.dot(s, x[mask].T)
            return ret.T
        else:
            raise ValueError(f"Cannot handle input worker label array of shape {X.shape}")

    def _step_expect_dense(self, X, y_true, sample_weight=None):
        """
        given true labels and workers' predictions, find confusion matrices

        Args:
            X: (n_workers, n_examples) matrix of worker prediction
                or (n_workers, n_examples, n_classes) array of predicted probabilities
            y_true: (n_examples,) or (n_examples, n_classes)
            n_classes: integer

        Returns:
            (n_workers, n_classes, n_classes)
        """
        return np.array([confusion_matrix(
            y_true=y_true, y_pred=x, n_classes=self.n_classes, normalize=self.normalize, sample_weight=sample_weight
        ) for x in X])

    def _step_expect_nan(self, X, y_true, sample_weight, notnull):
        """
        Args:
            X: (n_workers, n_examples) or (n_workers, n_examples, n_classes)
            y_true: (n_examples,) or (n_examples, n_classes)
            sample_weight: (n_examples,)
            n_classes: int
            notnull: (n_workers, n_examples)
        Returns:
            (n_workers, n_classes, n_classes)
        """
        cm_list = []
        for x, mask in zip(X, notnull):
            cm_list.append(confusion_matrix(
                y_true=y_true[mask], y_pred=x[mask], sample_weight=sample_weight[mask],
                n_classes=self.n_classes, normalize=self.normalize))
        return np.array(cm_list)

    def fit_dense(self, X, y, sample_weight=None):
        # todo: should gold matter after initialization?
        self.worker_confusion_matrices = self._step_expect_dense(X, y_true=y, sample_weight=sample_weight)

    def predict_proba(self, X, sample_weight=None):
        """
        X_gold.shape[0] == X.shape[0]  # the same pool of workers
        """
        if sparse.issparse(X):
            raise NotImplementedError  # TODO: handle sparse matrices as graphs
        else:
            notnull = np.isfinite(X)  # 2d or 3d array
            if notnull.all():  # dense array
                _step_maximize = lambda S: EMClassifier._step_maximize_dense(X, S)
                _step_expect = lambda y_pred: self._step_expect_dense(X, y_pred, sample_weight=sample_weight)
            else:  # array with NaN
                if X.ndim == 3:  # array stores predicted probability
                    notnull = notnull.all(axis=-1)  # will now be shape (n_workers, n_examples)
                _step_maximize = lambda S: EMClassifier._step_maximize_nan(X, S, notnull=notnull)
                _step_expect = lambda y_pred: self._step_expect_nan(X, y_pred, sample_weight=sample_weight, notnull=notnull)

        if self.worker_confusion_matrices is None:  # no gold standard; initialize true labels by majority voting
            y_old = stats.mode(X, axis=0, nan_policy="omit").mode.flatten()
        else:  # initialize true labels by weighted votes (from gold labels)
            y_old = _step_maximize(self.worker_confusion_matrices).argmax(axis=1)
        for _ in range(self.max_iter):
            s = _step_expect(y_old)
            y_new_prob = _step_maximize(s)
            y_new = y_new_prob.argmax(axis=1)
            if (y_old != y_new).sum() <= y_old.shape[0] * self.tol:  # converges
                self.worker_confusion_matrices = s
                return array_normalize(y_new_prob, axis=1)
            else:
                y_old = y_new
        s = _step_expect(y_old)
        self.worker_confusion_matrices = s
        return array_normalize(_step_maximize(s), axis=1)

    def get_worker_confuion_matrice(self):
        return self.worker_confusion_matrices

# def naive_em_classify(A, l, max_iter):
#     # initialize true labels by majority voting
#     # TODO: is EM sensitive to initialization? randomized tie-breaking?
#     y = stats.mode(A, axis=0, nan_policy="omit")[0][0]
#     for _ in range(max_iter): # until convergence
#         # normalize confusion matrix by predicted class
#         s = np.array([confusion_matrix(y, row, l, normalize=True, normalization_axis=0) for row in A])
#         y_new = sum(s_mat[:, a_row] for a_row, s_mat in zip(A, s)).argmax(axis=0)
#         if np.array_equal(y_new, y): # convergence
#             return y, s
#         y = y_new
#     return y, s

