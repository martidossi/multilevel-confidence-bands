import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


ArrayLike = Union[np.ndarray, pd.Series]

def model_eval_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict:
    """
    Compute standard regression evaluation metrics.

    Returns
    -------
    dict
        - mse  : Mean Squared Error
        - rmse : Root Mean Squared Error
        - mape : Mean Absolute Percentage Error (in %)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape #round(mape, 2),
    }

    return metrics