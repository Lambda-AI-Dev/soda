import numpy as np
import numba

from ..utils import array_normalize


@numba.jit
def _cm_11(y1, y2, n_classes, sample_weight=None):
    """
    Args:
        y1: (n_examples, ) array of class indices
        y2: (n_examples, ) array of class indices
        n_classes: total number of disjoint classes

    Returns:
        cm: (possibly weighted) counts
            cm[j, k] denotes the number of elements i such that y1[i]==j and y2[i]==k
    """
    cm = np.zeros((n_classes, n_classes))
    n_examples = y1.shape[0]
    if sample_weight is None:
        for i in range(n_examples):
            cm[y1[i], y2[i]] += 1
    else:
        for i in range(n_examples):
            cm[y1[i], y2[i]] += sample_weight[i]
    return cm


@numba.jit
def _cm_12(y1, y2, n_classes, sample_weight=None):
    cm = np.zeros((n_classes, n_classes))
    n_examples = y1.shape[0]
    if sample_weight is None:
        for i in range(n_examples):
            # y1[i] is the class in y1; y2[i] is the probability vector in y2
            cm[y1[i]] += y2[i]  # increment by probability vector instead
    else:
        for i in range(n_examples):
            cm[y1[i]] += y2[i] * sample_weight[i]
    return cm


def _cm_22(arr_1: np.ndarray, arr_2: np.ndarray, sample_weight=None):
    """
    both arrays have shape (n_examples, n_classes)
    """
    if sample_weight is None:
        return arr_1.T.dot(arr_2)
    else:
        return (arr_1.T * sample_weight).dot(arr_2)  # use matrix algebra to simplify expression


def normalize_confusion_matrix(cm, normalize=None):
    if normalize is None:
        return cm
    else:
        axis_dict = {
            "pred": 0,  # divide by the sum of column
            "true": 1,  # divide by the sum of row
            "all": None  # divide by the sum of all counts
        }
        # if normalize is not None and also not expected, automatically gives "all"
        return array_normalize(cm, axis=axis_dict.setdefault(normalize))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, sample_weight=None, normalize=None):
    """
    Important Note: It's assumed that the input doesn't contain any missing values

    Generalized confusion matrix

    About normalization axis:
    normalization_axis (int): if normalize is True, which axis to normalize along:
        Notice the invariant that cm must be a 2d square matrix.
        0: Useful for conditional probability of true_class = i, given predicted_class = j (column query)
        1: Useful for conditional probability of predicted_class = j, given true_class = i (row query)
        None: Useful for joint probability of true_class = i and predicted_class = j.
        None will be used when the normalization axis is invalid.

    Parameters
    ----------
    y_true : array-like of shape (n_examples,) or (n_examples, n_classes)
    (n_examples, ) array of true class labels
        or (n_examples, n_classes) array of soft labels (class probabilities)

    y_pred : (n_examples, ) array of predicted class labels
        or (n_examples, n_classes) array of predicted class probabilities
        The class probabilities must be normalized

    n_classes : number of unique class labels (from 0 to n_classes-1, inclusive)

    sample_weight : (n_examples, ) array of sample weight for each example

    normalize : one of {'true', 'pred', 'all', None} how to normalize the confusion matrix
        if None, then the count is returned

    Returns
    -------
    cm : (n_classes, n_classes) confusion matrix of counts/frequency. (truth * predicted)
        The vertical axis (axis 0) is truth and the horizontal axis (axis 1) is predicted.
        cm[i, j] counts the number of instances whose actual class is i and predicted class j.
    """
    assert isinstance(y_true, np.ndarray)

    cm = None
    n_examples = y_true.shape[0]

    if sample_weight is not None and sample_weight.shape != (n_examples,):
        raise ValueError(f"invalid shapes: y_true.shape = {y_true.shape}, y_pred.shape = {y_pred.shape},"
                         f"sample_weight.shape = {sample_weight.shape}")

    if y_true.shape == (n_examples,):
        if y_pred.shape == (n_examples, ):
            cm = _cm_11(y_true, y_pred, n_classes, sample_weight)
        elif y_pred.shape == (n_examples, n_classes):
            cm = _cm_12(y_true, y_pred, n_classes, sample_weight)
    elif y_true.shape == (n_examples, n_classes):
        if y_pred.shape == (n_examples, ):
            # switch y_pred and y_true for code reuse; cm(y2, y1).T = cm(y1, y2)
            cm = _cm_12(y_pred, y_true, n_classes, sample_weight).T
        elif y_pred.shape == (n_examples, n_classes):
            cm = _cm_22(y_true, y_pred, sample_weight)

    if cm is None:
        raise ValueError(f"invalid shapes: y_true.shape = {y_true.shape}, y_pred.shape = {y_pred.shape}")

    return normalize_confusion_matrix(cm, normalize=normalize)


