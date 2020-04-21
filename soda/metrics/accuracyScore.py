import numpy as np


def _weighted_sum(a: np.ndarray, w: np.ndarray = None):
    if w is None:
        return a.sum()
    else:
        return (a * w).sum()


def _acc_11(y1: np.ndarray, y2: np.ndarray, sample_weight=None):
    """
    accuracy counts. If two labels match, then count += 1
    """
    b = y1 == y2
    return _weighted_sum(b, sample_weight)


def _acc_12(y1: np.ndarray, y2: np.ndarray, sample_weight=None):
    """
    accuracy counts. If y1[i] gives label j and y2[i, j] is probability p, then return p
    """
    b = y2[np.arange(y2.shape[0]), y1]
    return _weighted_sum(b, sample_weight)


def _acc_22(y1: np.ndarray, y2: np.ndarray, sample_weight=None):
    """
    generalized accuracy counts. This is equivalent to the sum of diagonal in a generalized confusion matrix
    """
    b = (y1 * y2).sum(axis=1)
    return _weighted_sum(b, sample_weight)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None, normalize=True):
    """
    generalized accuracy score for 2 dimensions as well
    """
    count = None
    n_examples = y_true.shape[0]

    if sample_weight is not None and sample_weight.shape != (n_examples,):
        raise ValueError(f"invalid shapes: y_true.shape = {y_true.shape}, y_pred.shape = {y_pred.shape},"
                         f"sample_weight.shape = {sample_weight.shape}")

    if y_true.ndim == 1:
        if y_pred.ndim == 1:
            count = _acc_11(y_true, y_pred, sample_weight)
        elif y_pred.ndim == 2:
            count = _acc_12(y_true, y_pred, sample_weight)
    elif y_true.ndim == 2:
        if y_pred.ndim == 1:
            # switch y_pred and y_true for code reuse
            count = _acc_12(y_pred, y_true, sample_weight)
        elif y_pred.ndim == 2:
            count = _acc_22(y_true, y_pred, sample_weight)

    if normalize:
        if sample_weight is None:
            return count / sample_weight.shape[0]
        else:
            return count / sample_weight.sum()
    else:
        return count
