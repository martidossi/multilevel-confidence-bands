from typing import Any, Dict, List, Union, Type
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from xgboost import XGBRegressor

from src import (
    get_data_sample,
    model_eval_metrics,
    evaluate_intervals,
    get_confidence_interval,
    get_conformalized_interval
)

def methods_simulation_study(
    dataset, 
    model_class: Type,
    interval_methods: Union[List[str], Dict[Type, List[str]]],
    n_iter: int,
    alpha: float = 0.05,
    train_size: float = 0.6,
    calib_size: float = 0.2,
    fixed_test: bool = True,
) -> Dict[str, Any]:
    """
    Run a simulation study to evaluate predictive performance and uncertainty
    intervals for a given model class and set of interval methods.

    Parameters
    ----------
    dataset :
        Sklearn-like dataset object with attributes ``data``, ``target``,
        and ``feature_names`` (e.g. from ``load_diabetes()``).
    model_class : type
        One of the supported regression model classes:
        ``RandomForestRegressor``, ``ExplainableBoostingRegressor``,
        or ``XGBRegressor``.
    interval_methods : list or dict
        If a list, specifies which interval methods to run
        (e.g. ``["pi", "conformal", "built-in"]``) for all models.
        If a dict, it should map model classes to lists:
        ``{RandomForestRegressor: [...], ExplainableBoostingRegressor: [...], ...}``.
    n_iter : int
        Number of random splits (simulation iterations).
    alpha : float, default=0.05
        Miscoverage level for (1 - alpha) prediction intervals.
    train_size : float, default=0.6
        Proportion of the data used for training.
    calib_size : float, default=0.2
        Proportion of the data used for calibration.
    fixed_test : bool, default=True
        If True, the test split is fixed across iterations (using a fixed random
        state inside ``get_data_sample``). If False, the test split also depends
        on the ``seed`` passed to ``get_data_sample``.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``data_props``         : proportions of train/calib/test for each seed
        - ``rmse``               : RMSE on train and test for each seed
        - ``pi_metrics``         : coverage/sharpness for standard PIs
        - ``conf_metrics``       : coverage/sharpness for conformalized PIs
        - ``bi_metrics``         : coverage/sharpness for built-in intervals
        - ``check_df_test``      : list of test set indices for each seed
    """
    # Initialization of containers
    data_props: Dict[str, Dict[str, float]] = {}
    rmse: Dict[str, Dict[str, float]] = {}
    pi_metrics: Dict[str, Dict[str, float]] = {}
    bi_metrics: Dict[str, Dict[str, float]] = {}
    conf_metrics: Dict[str, Dict[str, float]] = {}
    check_df_test: List[List[int]] = []

    z_score = norm.ppf(1 - alpha / 2.0)

    # Input validation: supported models
    VALID_MODELS = {RandomForestRegressor, ExplainableBoostingRegressor, XGBRegressor}
    if model_class not in VALID_MODELS:
        raise ValueError(
            f"Invalid model_class: {model_class}. "
            f"Must be one of {', '.join(cls.__name__ for cls in VALID_MODELS)}."
        )

    # Determine which interval methods to run
    if isinstance(interval_methods, list):
        methods_to_run = interval_methods
    elif isinstance(interval_methods, dict):
        methods_to_run = interval_methods.get(model_class, [])
    else:
        raise TypeError("interval_methods must be either a list or a dict.")

    # Build X, y from the dataset (sklearn-like)
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    n = len(y)

    # Main simulation loop
    for seed in tqdm(range(n_iter), desc="Simulation Progress"):
        key = f"seed_{seed}"

        # Data split: for every seed, train/calib change; test may be fixed or not
        np.random.seed(seed)

        data = get_data_sample(
            X=X,
            y=y,
            train_size=train_size,
            calib_size=calib_size,
            seed=seed,
            fixed_test=fixed_test,
        )  # type: ignore[assignment] if returning DataSplit

        # If get_data_sample returns a DataSplit dataclass (recommended):
        # data_props
        data_props[key] = {
            "train": data.X_train.shape[0] / n,
            "calib": data.X_calib.shape[0] / n,
            "test": data.X_test.shape[0] / n,
        }

        check_df_test.append(list(data.X_test.index))

        # ---------------------------------------------------------------------
        # Model initialization (single-threaded & deterministic where possible)
        # ---------------------------------------------------------------------
        if model_class == RandomForestRegressor:
            model = RandomForestRegressor(
                random_state=42,
                n_jobs=1,
            )
        elif model_class == ExplainableBoostingRegressor:
            model = ExplainableBoostingRegressor(
                random_state=42,
            )
        elif model_class == XGBRegressor:
            model = XGBRegressor(
                random_state=42,
                seed=42, # redundant but explicit
                nthread=1, # single-threaded for determinism
                deterministic_histogram=True,
                enable_categorical=False,
                verbosity=0,
            )

        # Fit model and get predictions
        model.fit(data.X_train, data.y_train)
        y_pred_train = model.predict(data.X_train)
        y_pred_calib = model.predict(data.X_calib)
        y_pred_test = model.predict(data.X_test)

        # Performance metrics (RMSE via helper)
        metrics_train = model_eval_metrics(data.y_train, y_pred_train)
        metrics_test = model_eval_metrics(data.y_test, y_pred_test)

        rmse[key] = {
            "train": metrics_train["rmse"],
            "test": metrics_test["rmse"],
        }

        # ---------------------------------------------------------------------
        # Uncertainty intervals
        # ---------------------------------------------------------------------

        # 1) Standard PIs (normal-based)
        if "pi" in methods_to_run:
            pi_lower_bound, pi_upper_bound = get_confidence_interval(
                y_test_pred=y_pred_test,
                y_train_pred=y_pred_train, 
                y_train=data.y_train,
                alpha=alpha,
            )
            pi_metrics[key] = evaluate_intervals(
                lower_bound=pi_lower_bound,
                upper_bound=pi_upper_bound,
                y_true=data.y_test,
            )

        # 2) Conformalized PIs
        if "conformal" in methods_to_run:
            conf_lower_bound, conf_upper_bound = get_conformalized_interval(
                y_test_pred=y_pred_test,
                y_calib_pred=y_pred_calib,
                y_calib=data.y_calib,
                alpha=alpha,
            )
            conf_metrics[key] = evaluate_intervals(
                lower_bound=conf_lower_bound,
                upper_bound=conf_upper_bound,
                y_true=data.y_test,
            )

        # 3) Built-in uncertainty (only for models that support it)
        if "built-in" in methods_to_run:
            if model_class == RandomForestRegressor:
                all_tree_preds = np.stack(
                    [tree.predict(np.array(data.X_test)) for tree in model.estimators_],
                    axis=0,
                )
                preds = all_tree_preds.mean(axis=0)
                uncertainties = all_tree_preds.std(axis=0)

            elif model_class == ExplainableBoostingRegressor:
                preds, uncertainties = model.predict_with_uncertainty(
                    data.X_test
                ).T

            else:
                # XGBRegressor does not provide a built-in predictive std here
                raise ValueError(
                    '"built-in" uncertainty is not supported for '
                    f"{model_class.__name__}."
                )

            bi_lower_bound = preds - z_score * uncertainties
            bi_upper_bound = preds + z_score * uncertainties
            bi_metrics[key] = evaluate_intervals(
                lower_bound=bi_lower_bound,
                upper_bound=bi_upper_bound,
                y_true=data.y_test,
            )


    return {
        "data_props": data_props,
        "rmse": rmse,
        "pi_metrics": pi_metrics,
        "conf_metrics": conf_metrics,
        "bi_metrics": bi_metrics,
        "check_df_test": check_df_test,
    }



def coverage_simulation_study(
    dataset,
    n_iter: int,
    alpha_list: List[float] = None,
    train_size: float = 0.6,
    calib_size: float = 0.2,
    fixed_test: bool = True,
) -> Dict[str, Any]:
    """
    Simulation study to evaluate conformal prediction intervals for
    different miscoverage levels (alpha).

    Parameters
    ----------
    dataset :
        Sklearn-like dataset object with attributes ``data``,
        ``target`` and ``feature_names``.
    n_iter : int
        Number of simulation iterations.
    alpha_list : list of float, optional
        Miscoverage levels to evaluate. Each alpha generates a
        (1 - alpha) conformal interval.
        Default: [0.001, 0.01, 0.05, 0.1, 0.2].
    train_size : float, default=0.6
        Proportion of dataset used for training.
    calib_size : float, default=0.2
        Proportion used for calibration.
    fixed_test : bool, default=True
        If True, the test split is fixed across all iterations.

    Returns
    -------
    dict
        Mapping (seed, alpha) → interval metrics (coverage, sharpness).
    """
    if alpha_list is None:
        alpha_list = [0.001, 0.01, 0.05, 0.1, 0.2]

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target

    conf_metrics: Dict[str, Any] = {}

    for seed in tqdm(range(n_iter), desc="Simulation Progress"):
        key = f"seed_{seed}"

        # Data split (using DataSplit)
        data = get_data_sample(
            X=X,
            y=y,
            train_size=train_size,
            calib_size=calib_size,
            seed=seed,
            fixed_test=fixed_test
        )  # returns DataSplit

        # Model
        model = ExplainableBoostingRegressor(random_state=42)
        model.fit(data.X_train, data.y_train)

        # Predictions
        y_pred_calib = model.predict(data.X_calib)
        y_pred_test = model.predict(data.X_test)

        # Evaluate all alpha levels
        for alpha in alpha_list:
            lower, upper = get_conformalized_interval(
                y_test_pred=y_pred_test,
                y_calib_pred=y_pred_calib,
                y_calib=data.y_calib,
                alpha=alpha,
            )
            conf_metrics[f"{key}_alpha_{alpha}"] = evaluate_intervals(
                lower_bound=lower,
                upper_bound=upper,
                y_true=data.y_test
            )

    return conf_metrics
