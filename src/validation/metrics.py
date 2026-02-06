import numpy as np


def squared_log_error(y_true, y_pred):
    """
    Computes the squared log error element-wise.
    y_pred is clipped to 0.
    Returns array of (log1p(pred) - log1p(true))^2.
    """
    y_pred = np.maximum(y_pred, 0)
    return np.square(np.log1p(y_pred) - np.log1p(y_true))


def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error.
    """
    return np.sqrt(np.mean(squared_log_error(y_true, y_pred)))
