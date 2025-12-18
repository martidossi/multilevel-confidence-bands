import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for train / calibration / test splits."""
    X_train: pd.DataFrame
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray]
    X_calib: pd.DataFrame
    y_calib: Union[pd.Series, pd.DataFrame, np.ndarray]
    X_test: pd.DataFrame
    y_test: Union[pd.Series, pd.DataFrame, np.ndarray]
    
    def __getitem__(self, key):
        return getattr(self, key)
    
def get_data_sample(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    seed: int,
    train_size: float,
    calib_size: float,
    fixed_test: bool = True,
) -> DataSplit:
    
    """
    Split the data into training, calibration, and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target vector or DataFrame with targets. Length must match ``X``.
    seed : int
        Random seed used for the splits (except the fixed test split).
    train_size : float
        Proportion of the whole dataset to use for training (0 < train_size < 1).
    calib_size : float
        Proportion of the whole dataset to use for calibration.
        If ``fixed_test`` is False, this is interpreted relative to the
        remaining data after training.
    fixed_test : bool, default=True
        If True, the test split is fixed using ``random_state=42``.
        If False, the test split also depends on the ``seed`` passed to the function.
    
    Returns
    -------
    DataSplit
        Dataclass containing ``X_train``, ``y_train``, ``X_calib``,
        ``y_calib``, ``X_test``, and ``y_test``.
    """

    # Check
    if train_size + calib_size >= 1.0:
        raise ValueError("train_size + calib_size must be < 1.")

    test_size = 1 - train_size - calib_size
    train_frac = train_size / (train_size + calib_size)
    random_seed_test = 42 if fixed_test else seed
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed_test)
    X_train, X_calib, y_train, y_calib = train_test_split(X_pool, y_pool, train_size=train_frac, random_state=seed)

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test,
        y_test=y_test,
    )
