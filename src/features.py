from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config

EARTH_RADIUS_MILES = 3958.8
# Cities are represented by coordinates and lane/city effects rather than as
# categories, because validation contains cities that never appear in training.
CATEGORICAL_FEATURES = ["equipment"]

TREND_ORIGIN = pd.Timestamp("2025-01-01")


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def _lane_key(frame: pd.DataFrame) -> pd.Series:
    return frame["pickup"].astype(str) + "|" + frame["delivery"].astype(str)


class DistanceCurve:
    """Expected log rate per mile from distance alone: a quadratic in log distance per equipment."""

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "DistanceCurve":
        log_distance = np.log(frame["distance"])
        self.fallback = np.polyfit(log_distance, target, 2)
        self.coefficients = {
            equipment: np.polyfit(log_distance.loc[group.index], target.loc[group.index], 2)
            for equipment, group in frame.groupby("equipment")
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_distance = np.log(frame["distance"].to_numpy())
        equipment = frame["equipment"].to_numpy()
        result = np.polyval(self.fallback, log_distance)
        for level, coefficients in self.coefficients.items():
            mask = equipment == level
            result[mask] = np.polyval(coefficients, log_distance[mask])
        return result


@dataclass
class LaneEncoder:
    """How much more or less each lane, pickup city and delivery city pays than the
    distance curve predicts (smoothed average residual of log rate per mile).

    A city that never appears in training borrows the inverse-distance-weighted
    average of its nearest known cities; an unseen lane falls back to the sum of its
    pickup and delivery effects. Training rows get out-of-fold values so a row never
    sees its own label.
    """

    cities: pd.DataFrame
    smoothing: float = 20.0
    n_splits: int = 5
    neighbours: int = 3
    tables: dict[str, pd.Series] = field(default_factory=dict)

    def _fit_tables(self, frame: pd.DataFrame, residual: pd.Series) -> dict[str, pd.Series]:
        keys = {"lane": _lane_key(frame), "pickup": frame["pickup"], "delivery": frame["delivery"]}
        tables = {}
        for name, key in keys.items():
            stats = residual.groupby(key).agg(["sum", "count"])
            tables[name] = stats["sum"] / (stats["count"] + self.smoothing)
        return tables

    def _city_effect(self, cities: pd.Series, table: pd.Series) -> np.ndarray:
        effect = cities.map(table)
        unseen = sorted(set(cities[effect.isna()]))
        if unseen:
            known = self.cities.loc[table.index]
            borrowed = {}
            for city in unseen:
                lat, lon = self.cities.loc[city, ["lat", "lon"]]
                miles = haversine_miles(lat, lon, known["lat"].to_numpy(), known["lon"].to_numpy())
                nearest = np.argsort(miles)[: self.neighbours]
                weights = 1.0 / np.maximum(miles[nearest], 1.0)
                borrowed[city] = float(np.average(table.to_numpy()[nearest], weights=weights))
            effect = effect.fillna(cities.map(borrowed))
        return effect.to_numpy()

    def _apply(self, frame: pd.DataFrame, curve: np.ndarray, tables: dict[str, pd.Series]) -> pd.DataFrame:
        encoded = pd.DataFrame(index=frame.index)
        encoded["curve_log_rpm"] = curve
        encoded["pickup_effect"] = self._city_effect(frame["pickup"], tables["pickup"])
        encoded["delivery_effect"] = self._city_effect(frame["delivery"], tables["delivery"])
        lane = _lane_key(frame).map(tables["lane"])
        encoded["lane_effect"] = lane.fillna(encoded["pickup_effect"] + encoded["delivery_effect"])
        encoded["lane_log_rpm"] = encoded["curve_log_rpm"] + encoded["lane_effect"]
        return encoded

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.curve = DistanceCurve().fit(frame, target)
        curve = self.curve.predict(frame)
        residual = target - curve
        self.tables = self._fit_tables(frame, residual)
        parts = []
        folds = KFold(self.n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        for fit_idx, apply_idx in folds.split(frame):
            tables = self._fit_tables(frame.iloc[fit_idx], residual.iloc[fit_idx])
            parts.append(self._apply(frame.iloc[apply_idx], curve[apply_idx], tables))
        return pd.concat(parts).loc[frame.index]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._apply(frame, self.curve.predict(frame), self.tables)


def quote_regimes(reference_loads: pd.DataFrame, expected_log_rpm: np.ndarray) -> pd.DataFrame:
    """Classify each day's quote_signal behaviour without using labels.

    Per day, correlate log(quote_signal) with a quote-free model's expected log rate
    per mile. quote_signal switches between three regimes: it tracks the rate
    (correlation ~ +0.97), mirrors it inversely (~ -0.95), or is unrelated noise (~ 0).
    """
    frame = pd.DataFrame({
        "date": reference_loads["date"].to_numpy(),
        "log_quote": np.log(reference_loads["quote_signal"].to_numpy()),
        "expected": expected_log_rpm,
    })
    correlation = frame.groupby("date")[["log_quote", "expected"]].corr().xs("log_quote", level=1)["expected"]
    regime = np.select([correlation > 0.5, correlation < -0.5], ["tracks rate", "inverted"], "noise")
    return pd.DataFrame({"correlation": correlation, "regime": regime})


def build_base_features(
    frame: pd.DataFrame, daily_market: pd.DataFrame, categories: dict[str, list[str]]
) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)

    features["distance"] = frame["distance"]
    features["log_distance"] = np.log(frame["distance"])
    straight_line = haversine_miles(
        frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"]
    )
    features["straight_line_miles"] = straight_line
    features["circuity"] = frame["distance"] / np.maximum(straight_line, 1.0)

    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        features[column] = frame[column]
    features["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    features["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]

    features["weight"] = frame["weight"]
    features["log_weight"] = np.log(frame["weight"])
    features["weight_was_negative"] = frame.get("weight_was_negative", 0)

    features["market_index"] = frame["market_index"]
    daily = daily_market.reindex(frame["date"]).set_index(frame.index)
    for column in ("daily_market_index", "market_index_7d", "market_index_28d", "market_momentum"):
        features[column] = daily[column]
    features["market_index_vs_day"] = frame["market_index"] - daily["daily_market_index"]

    features["days_since_start"] = (frame["date"] - TREND_ORIGIN).dt.days
    features["day_of_week"] = frame["date"].dt.dayofweek
    features["day_of_month"] = frame["date"].dt.day
    for column in CATEGORICAL_FEATURES:
        features[column] = pd.Categorical(frame[column], categories=categories[column])
    return features


def category_levels(*frames: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(set().union(*(frame[column].unique() for frame in frames)))
        for column in CATEGORICAL_FEATURES
    }


def log_rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame[config.TARGET] / frame["distance"])
