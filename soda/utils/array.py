import numpy as np


def array_normalize(arr: np.ndarray, axis=None):
    """
    Normalize a numpy array by a certain axis (or collection of axes).
    This function will not create additional NaNs, because division by zero is avoided
    and the existing NaNs won't propagate to the sum.

    Args:
    arr: the array to normalize
    axis: {int, tuple of int, None}, optional

    Returns:
    The normalized array.
    """
    s = np.nansum(arr, axis=axis, keepdims=True)  # s will always be a np.ndarray due to keepdims=True
    s[s == 0] = 1  # prevent nan (division by 0)
    return arr / s
