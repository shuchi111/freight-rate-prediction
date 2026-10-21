from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .models import RateModel

ModelFactory = Callable[[], RateModel]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE_%": float(np.mean(np.abs(error) / actual) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def evaluate_split(
    factory: ModelFactory, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[RateModel, pd.Series, dict[str, float], dict[str, float]]:
    """Fit on clean training rows; score on all test rows and on clean test rows."""
    model = factory().fit(train[~train["label_outlier"]])
    predicted = pd.Series(model.predict(test), index=test.index)
    all_rows = regression_metrics(test[config.TARGET], predicted)
    clean = ~test["label_outlier"]
    clean_rows = regression_metrics(test.loc[clean, config.TARGET], predicted[clean])
    return model, predicted, all_rows, clean_rows


def holdout_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(config.HOLDOUT_START)
    return frame[frame["date"] < cutoff], frame[frame["date"] >= cutoff]


def new_city_holdout(
    factory: ModelFactory, frame: pd.DataFrame, n_cities: int = 8, seed: int = config.RANDOM_SEED
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Time holdout where some cities are also removed from training entirely.

    The prediction set contains cities that never appear in the labeled data, so this
    measures how well the model prices loads touching a brand-new city.
    """
    cities = sorted(set(frame["pickup"]) | set(frame["delivery"]))
    held = sorted(np.random.default_rng(seed).choice(cities, n_cities, replace=False))
    touches_held = frame["pickup"].isin(held) | frame["delivery"].isin(held)
    train, test = holdout_split(frame)
    train = train[~touches_held.loc[train.index]]
    model = factory().fit(train[~train["label_outlier"]])
    test = test[~test["label_outlier"]]
    predicted = model.predict(test)
    new_city = touches_held.loc[test.index].to_numpy()
    results = {
        "loads touching a new city": regression_metrics(test[config.TARGET][new_city], predicted[new_city]),
        "loads between known cities": regression_metrics(test[config.TARGET][~new_city], predicted[~new_city]),
    }
    return held, results


def rolling_splits(frame: pd.DataFrame):
    """Expanding window: train on all months before the test month."""
    month = frame["date"].dt.to_period("M").astype(str)
    for test_month in config.ROLLING_TEST_MONTHS:
        yield test_month, frame[month < test_month], frame[month == test_month]


def rolling_cv(factory: ModelFactory, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_month, train, test in rolling_splits(frame):
        _, _, _, clean = evaluate_split(factory, train, test)
        rows.append({"test_month": test_month, "train_rows": len(train), "test_rows": len(test), **clean})
    return pd.DataFrame(rows)


def error_breakdown(test: pd.DataFrame, predicted: pd.Series, seen_lanes: set) -> pd.DataFrame:
    clean = test[~test["label_outlier"]].copy()
    clean["abs_pct_error"] = (predicted[clean.index] - clean[config.TARGET]).abs() / clean[config.TARGET] * 100
    clean["abs_error"] = (predicted[clean.index] - clean[config.TARGET]).abs()
    lanes = list(zip(clean["pickup"], clean["delivery"]))
    groups = {
        "equipment": clean["equipment"],
        "distance band": pd.cut(clean["distance"], [0, 300, 800, 1500, np.inf],
                                labels=["<300 mi", "300-800 mi", "800-1500 mi", ">1500 mi"]),
        "lane seen in training": pd.Series(
            np.where([lane in seen_lanes for lane in lanes], "seen", "unseen"), index=clean.index
        ),
    }
    tables = []
    for name, key in groups.items():
        table = clean.groupby(key, observed=True).agg(
            rows=("abs_error", "size"), MAE=("abs_error", "mean"), MAPE_pct=("abs_pct_error", "mean")
        )
        table.index = [f"{name}: {value}" for value in table.index]
        tables.append(table)
    return pd.concat(tables).round(2)
