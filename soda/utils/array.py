import numpy as np
import pandas as pd


def array_normalize(arr, axis=None):
    """
    Normalize a numpy array by a certain axis (or collection of axes).
    This function will not create additional NaNs, because division by zero is avoided
    and the existing NaNs won't propagate to the sum.

    If the input is a pandas DataFrame, the index and the columns from the input is preserved

    Args:
    arr: the array to normalize
    axis: {int, tuple of int, None}, optional

    Returns:
    The normalized array.
    """
    if isinstance(arr, pd.DataFrame):
        ret_values = array_normalize(arr.values, axis=axis)
        return pd.DataFrame(ret_values, columns=arr.columns, index=arr.index)
    elif isinstance(arr, np.ndarray):
        s = np.nansum(arr, axis=axis, keepdims=True)  # s will always be a np.ndarray due to keepdims=True
        s[s == 0] = 1  # prevent nan (division by 0)
        return arr / s
    else:
        raise ValueError(f"Unsupported array type: {type(arr)}")
