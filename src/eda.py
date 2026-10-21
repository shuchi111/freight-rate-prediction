"""Exploration figures used in the report and notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

COLOR = "#064A56"
ACCENT = "#D9822B"


def _style(axis, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def rate_per_mile_by_distance(train: pd.DataFrame, path: Path) -> None:
    clean = train[~train["label_outlier"]]
    rpm = clean[config.TARGET] / clean["distance"]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    for equipment, group in clean.groupby("equipment"):
        bands = pd.cut(group["distance"], [0, 150, 300, 500, 800, 1200, 2000, 3500])
        medians = (group[config.TARGET] / group["distance"]).groupby(bands, observed=True).median()
        axis.plot([b.mid for b in medians.index], medians.values, marker="o", label=equipment)
    axis.legend(frameon=False)
    _style(axis, "Rate per mile falls with distance and differs by equipment", "Distance (miles)", "Median $ per mile")
    axis.set_ylim(rpm.quantile(0.01), rpm.quantile(0.99))
    _save(figure, path)


def label_outliers(train: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    normal = train[~train["label_outlier"]]
    bad = train[train["label_outlier"]]
    axis.scatter(normal["distance"], normal[config.TARGET], s=2, alpha=0.15, color=COLOR, label="normal")
    axis.scatter(bad["distance"], bad[config.TARGET], s=6, alpha=0.7, color=ACCENT,
                 label=f"flagged label errors ({len(bad)})")
    axis.set_yscale("log")
    axis.legend(frameon=False, markerscale=3)
    share = len(bad) / len(train) * 100
    _style(axis, f"{share:.1f}% of labels are far above or below normal", "Distance (miles)", "Posted rate ($, log scale)")
    _save(figure, path)


def quote_regimes(regimes: pd.DataFrame, path: Path) -> None:
    colors = {"tracks rate": COLOR, "inverted": ACCENT, "noise": "#9DAFB3"}
    figure, axis = plt.subplots(figsize=(10, 4.2))
    for regime, group in regimes.groupby("regime"):
        axis.scatter(group.index, group["correlation"], s=8, color=colors[regime], label=regime)
    axis.axhline(0, color="#9DAFB3", linewidth=0.8)
    axis.axvline(pd.Timestamp("2025-11-01"), color="#455A60", linestyle="--", linewidth=1)
    axis.text(pd.Timestamp("2025-11-03"), 0.75, "prediction\nperiod", fontsize=9, color="#455A60")
    axis.legend(frameon=False, loc="lower left", fontsize=9)
    _style(axis, "quote_signal switches between tracking, inverting and ignoring the rate",
           "", "Daily corr(log quote, expected log rate)")
    _save(figure, path)


def residual_trend(train: pd.DataFrame, path: Path) -> pd.Series:
    clean = train[~train["label_outlier"]].copy()
    clean["log_rpm"] = np.log(clean[config.TARGET] / clean["distance"])
    clean["residual"] = clean["log_rpm"] - clean.groupby(["pickup", "delivery", "equipment"])["log_rpm"].transform("mean")
    design = np.column_stack([np.ones(len(clean)), clean["market_index"], np.log(clean["weight"])])
    coefficients, *_ = np.linalg.lstsq(design, clean["residual"], rcond=None)
    clean["unexplained"] = clean["residual"] - design @ coefficients
    weekly = clean.groupby(pd.Grouper(key="date", freq="W"))["unexplained"].mean() * 100
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.plot(weekly.index, weekly.values, color=COLOR, marker="o", markersize=3)
    axis.axhline(0, color="#9DAFB3", linewidth=0.8)
    _style(axis, "Rates drift upward beyond what market_index explains",
           "", "Lane-adjusted rate vs average (%)")
    _save(figure, path)
    return weekly


def model_comparison(holdout: pd.DataFrame, rolling: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    pivot = rolling.pivot(index="test_month", columns="model", values="MAPE_%")
    for model in pivot.columns:
        axis.plot(pivot.index, pivot[model], marker="o", label=model)
    axis.legend(frameon=False)
    _style(axis, "Rolling monthly validation (train on all earlier months)", "Test month", "MAPE (%)")
    _save(figure, path)


def december_inputs(december: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(december["date"], december["market_index"], color=COLOR, marker="o", markersize=3)
    _style(axes[0], "Daily market_index used for December", "", "market_index")
    axes[1].plot(december["date"], december["predicted_rate"], color=ACCENT, marker="o", markersize=3)
    _style(axes[1], "Predicted rate, Lexington to Fort Wayne", "", "$")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    _save(figure, path)
