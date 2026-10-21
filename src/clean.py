from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

DISTANCE_BANDS = [0, 150, 300, 500, 800, 1200, 2000, np.inf]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])
    validation = pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"])
    december = pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])
    return train, validation, december


def build_city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its coordinates, taken from pickup and delivery columns."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            part = frame[[side, f"{side}_lat", f"{side}_lon"]]
            part.columns = ["city", "lat", "lon"]
            parts.append(part)
    return pd.concat(parts).groupby("city")[["lat", "lon"]].median()


def build_daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily average market_index across all loads posted that day.

    Uses only input features (never labels), so validation rows can contribute.
    Gaps in the calendar are filled by time interpolation.
    """
    combined = pd.concat([frame[["date", "market_index"]] for frame in frames])
    daily = combined.groupby("date").agg(
        daily_market_index=("market_index", "mean"),
        daily_load_count=("market_index", "size"),
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar).interpolate(method="time").ffill().bfill()
    daily.index.name = "date"
    daily["market_index_7d"] = daily["daily_market_index"].rolling(7, min_periods=1).mean()
    daily["market_index_28d"] = daily["daily_market_index"].rolling(28, min_periods=1).mean()
    daily["market_momentum"] = daily["market_index_7d"] - daily["market_index_28d"]
    return daily


@dataclass
class FeatureCleaner:
    """Fixes input-feature quality issues. Fitted on training rows only."""

    weight_medians: dict[str, float] = field(default_factory=dict)
    overall_weight_median: float = 0.0

    def fit(self, frame: pd.DataFrame) -> "FeatureCleaner":
        weight = frame["weight"].abs()
        self.weight_medians = weight.groupby(frame["equipment"]).median().to_dict()
        self.overall_weight_median = float(weight.median())
        return self

    def transform(self, frame: pd.DataFrame, daily_market: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()

        result["weight_was_negative"] = (result["weight"] < 0).astype(int)
        result["weight_was_missing"] = result["weight"].isna().astype(int)
        result["weight"] = result["weight"].abs()
        fallback = result["equipment"].map(self.weight_medians).fillna(self.overall_weight_median)
        result["weight"] = result["weight"].fillna(fallback)

        result["market_index_was_missing"] = result["market_index"].isna().astype(int)
        day_average = result["date"].map(daily_market["daily_market_index"])
        result["market_index"] = result["market_index"].fillna(day_average)
        return result


def flag_label_outliers(frame: pd.DataFrame) -> pd.Series:
    """True for rows whose posted_rate looks like a data-entry error.

    The bad labels sit at roughly 4x or 0.25x the normal rate per mile for the
    same equipment and distance band, far outside the natural spread.
    """
    rate_per_mile = frame[config.TARGET] / frame["distance"]
    band = pd.cut(frame["distance"], DISTANCE_BANDS)
    expected = rate_per_mile.groupby([frame["equipment"], band], observed=True).transform("median")
    ratio = rate_per_mile / expected
    return (ratio > config.OUTLIER_HIGH_RATIO) | (ratio < config.OUTLIER_LOW_RATIO)


def complete_december_inputs(
    december: pd.DataFrame, cities: pd.DataFrame, daily_market: pd.DataFrame
) -> pd.DataFrame:
    """Add the columns the model needs but the December chart file does not provide."""
    result = december.copy()
    for side in ("pickup", "delivery"):
        result[f"{side}_lat"] = result[side].map(cities["lat"])
        result[f"{side}_lon"] = result[side].map(cities["lon"])
    result["market_index"] = result["date"].map(daily_market["daily_market_index"])
    if result[["pickup_lat", "delivery_lat", "market_index"]].isna().any().any():
        raise ValueError("December inputs reference a city or date with no reference data")
    return result
