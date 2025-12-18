import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series]

def evaluate_intervals(
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    y_true: ArrayLike,
) -> dict:
    """
    Compute key evaluation metrics for Prediction Intervals (PIs).

    Parameters
    ----------
    lower_bound : array-like
        Lower endpoints of the PI.
    upper_bound : array-like
        Upper endpoints of the PI.
    y_true : array-like
        True target values.

    Returns
    -------
    dict
        - 'coverage'  : empirical coverage of the intervals
        - 'sharpness' : average interval width
    """
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    y_true = np.asarray(y_true)

    coverage = calculate_coverage(
        y_true=y_true,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    sharpness = float(np.mean(upper_bound - lower_bound))

    return {
        "coverage": coverage,
        "sharpness": sharpness,
    }

def calculate_coverage(
    y_true: ArrayLike,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Parameters
    ----------
    y_true : array-like
        True target values (NumPy array or pandas Series).
    lower_bound : array-like
        Lower endpoints of the prediction intervals.
    upper_bound : array-like
        Upper endpoints of the prediction intervals.

    Returns
    -------
    float
        Proportion of observations where ``y_true`` lies inside the interval.
    """
    y_true = np.asarray(y_true)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    return float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))


def get_confidence_interval(
    y_test_pred: ArrayLike,
    y_train_pred: ArrayLike,
    y_train: ArrayLike,
    alpha: float = 0.05,
    ddof: int = 1,
    bias_correct: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normal-based symmetric confidence (or prediction) intervals using training residuals.

    Parameters
    ----------
    y_test_pred : array-like
        Model predictions on the test set.
    y_train_pred : array-like
        Model predictions on the training set.
    y_train : array-like
        True target values on the training set.
    alpha : float, default=0.05
        Miscoverage level (0.05 → 95% confidence interval).
    ddof : int, default=1
        Degrees of freedom for the sample standard deviation.
        A value of 1 is typically recommended.
    bias_correct : bool, default=False
        If True, shift the interval center by the mean training residual
        (bias correction).

    Returns
    -------
    lower_bound : np.ndarray
        Lower endpoints of the confidence intervals.
    upper_bound : np.ndarray
        Upper endpoints of the confidence intervals.

    Notes
    -----
    - Intervals are symmetric around the (possibly bias-corrected) point prediction.
    - Residuals are used to estimate the standard deviation of the error.
    """
    
    # Convert inputs to NumPy arrays
    y_test_pred = np.asarray(y_test_pred)
    y_train_pred = np.asarray(y_train_pred)
    y_train = np.asarray(y_train)

    # Residuals (signed)
    residuals = y_train - y_train_pred

    # Standard deviation of residuals
    std_residuals = np.std(residuals, ddof=ddof)

    # Z-score for the (1 - alpha) confidence interval
    z_score = norm.ppf(1 - alpha / 2)

    # Optional bias correction
    if bias_correct:
        mean_resid = np.mean(residuals)
        center = y_test_pred + mean_resid
    else:
        center = y_test_pred

    # Symmetric confidence intervals
    lower_bound = center - z_score * std_residuals
    upper_bound = center + z_score * std_residuals

    return lower_bound, upper_bound

def get_conformalized_interval(
    y_test_pred: ArrayLike,
    y_calib_pred: ArrayLike,
    y_calib: ArrayLike,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric conformal PIs using calibration residuals.

    Parameters
    ----------
    y_test_pred : array-like
        Model predictions on the test set.
    y_calib_pred : array-like
        Model predictions on the calibration set.
    y_calib : array-like
        True target values on the calibration set.
    alpha : float, default=0.05
        Miscoverage level for the (1 - alpha) prediction intervals.

    Returns
    -------
    lower_bound : np.ndarray
        Lower endpoints of the conformal prediction intervals.
    upper_bound : np.ndarray
        Upper endpoints of the conformal prediction intervals.

    Notes
    -----
    This is the standard *split conformal regression* method:
    - nonconformity scores are absolute residuals on the calibration set
    - intervals are symmetric around the point predictions
    """
    y_calib = np.asarray(y_calib)
    y_calib_pred = np.asarray(y_calib_pred)
    y_test_pred = np.asarray(y_test_pred)

    # Absolute residuals (nonconformity scores), we care about the magnitude of the error, not its direction
    calib_residuals = np.abs(y_calib - y_calib_pred)

    # Quantile of residuals
    q_hat = np.quantile(calib_residuals, 1 - alpha)

    # Symmetric conformal interval
    lower_bound = y_test_pred - q_hat
    upper_bound = y_test_pred + q_hat

    return lower_bound, upper_bound