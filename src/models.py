from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .clean import DISTANCE_BANDS
from .features import CATEGORICAL_FEATURES, LaneEncoder, build_base_features, log_rate_per_mile

LGBM_PARAMS = {
    "objective": "huber",
    "alpha": 0.15,
    "learning_rate": 0.03,
    "n_estimators": 2500,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": config.RANDOM_SEED,
    "verbose": -1,
}

TREND_FEATURES = ["days_since_start", "market_index", "log_weight", "log_distance"]

# One value per calendar day: in a tree model these act like a date ID and let it
# memorise each training day's level, which does not carry over to new dates.
DAY_LEVEL_FEATURES = [
    "daily_market_index", "market_index_7d", "market_index_28d", "market_momentum",
    "day_of_month", "market_index_vs_day",
]


@dataclass
class ModelContext:
    """Label-free reference data shared by every model.

    reference_loads holds the input features of every known load (training and
    prediction period) so date-level signals exist for any prediction date.
    """

    daily_market: pd.DataFrame
    categories: dict[str, list[str]]
    cities: pd.DataFrame
    reference_loads: pd.DataFrame


class RateModel:
    """Base class: learns log(rate per mile) and returns rates in dollars."""

    name = "base"

    def __init__(self, context: ModelContext):
        self.context = context

    def fit(self, frame: pd.DataFrame) -> "RateModel":
        raise NotImplementedError

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_rpm(frame)) * frame["distance"].to_numpy()


class LaneAverageBaseline(RateModel):
    """Median rate per mile of the lane (or equipment x distance band) times distance."""

    name = "Baseline: lane average"

    def fit(self, frame: pd.DataFrame) -> "LaneAverageBaseline":
        target = log_rate_per_mile(frame)
        self.lane = target.groupby([frame["pickup"], frame["delivery"], frame["equipment"]]).median()
        self.band = target.groupby([frame["equipment"], self._band(frame)], observed=True).median()
        return self

    @staticmethod
    def _band(frame: pd.DataFrame) -> pd.Series:
        return pd.cut(frame["distance"], DISTANCE_BANDS)

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        lane_keys = pd.MultiIndex.from_frame(frame[["pickup", "delivery", "equipment"]])
        band_keys = pd.MultiIndex.from_arrays([frame["equipment"], self._band(frame)])
        lane = pd.Series(self.lane.reindex(lane_keys).to_numpy())
        band = pd.Series(self.band.reindex(band_keys).to_numpy())
        return lane.fillna(band).to_numpy()


class FeatureModel(RateModel):
    """Shared feature construction for the Ridge and LightGBM models."""

    def _features(self, frame: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
        base = build_base_features(frame, self.context.daily_market, self.context.categories)
        return pd.concat([base, lanes], axis=1)

    def fit(self, frame: pd.DataFrame) -> "FeatureModel":
        target = log_rate_per_mile(frame)
        self.encoder = LaneEncoder(self.context.cities)
        lanes = self.encoder.fit_transform(frame, target)
        features = self._features(frame, lanes)
        self.feature_names = list(features.columns)
        self._fit_estimator(features, target)
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        features = self._features(frame, self.encoder.transform(frame))
        return self._predict_estimator(features[self.feature_names])


def fit_ridge(features: pd.DataFrame, target: pd.Series):
    numeric = [c for c in features.columns if c not in CATEGORICAL_FEATURES]
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return make_pipeline(transformer, Ridge(alpha=1.0)).fit(features, target)


class RidgeModel(FeatureModel):
    name = "Ridge regression"

    def _fit_estimator(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.estimator = fit_ridge(features, target)

    def _predict_estimator(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(features)


class LightGBMModel(FeatureModel):
    """Gradient-boosted trees on log rate per mile.

    trend="linear" first fits a linear model with a time trend, so the slow upward
    drift in rates can be extrapolated past the last training date (trees alone
    would flatten it); the trees then learn what the linear model leaves over.
    """

    name = "LightGBM"

    def __init__(self, context: ModelContext, params: dict | None = None, trend: str | None = "linear",
                 drop_day_level: bool = True):
        super().__init__(context)
        self.params = {**LGBM_PARAMS, **(params or {})}
        self.trend = trend
        self.drop_day_level = drop_day_level

    def _trend_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = features[TREND_FEATURES].copy()
        for level in self.context.categories["equipment"]:
            matrix[f"equipment_{level}"] = (features["equipment"] == level).astype(float)
        return matrix

    def _tree_features(self, features: pd.DataFrame) -> pd.DataFrame:
        drop = [] if self.trend == "feature" else ["days_since_start"]
        if self.drop_day_level:
            drop += DAY_LEVEL_FEATURES
        if self.trend == "ridge":
            drop += ["market_index"]
        return features.drop(columns=drop)

    def _first_stage(self, features: pd.DataFrame) -> np.ndarray:
        if self.trend == "linear":
            return self.trend_model.predict(self._trend_matrix(features))
        if self.trend == "ridge":
            return self.trend_model.predict(features)
        return np.zeros(len(features))

    def _fit_estimator(self, features: pd.DataFrame, target: pd.Series) -> None:
        if self.trend == "linear":
            self.trend_model = LinearRegression().fit(self._trend_matrix(features), target)
        elif self.trend == "ridge":
            self.trend_model = fit_ridge(features, target)
        residual = target - self._first_stage(features)
        self.estimator = lgb.LGBMRegressor(**self.params)
        self.estimator.fit(self._tree_features(features), residual, categorical_feature=CATEGORICAL_FEATURES)

    def _predict_estimator(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._tree_features(features)) + self._first_stage(features)

    def trend_per_month(self) -> float:
        """Rate change per 30 days implied by the first-stage time coefficient, as a fraction."""
        if self.trend == "ridge":
            transformer, ridge = self.trend_model[0], self.trend_model[-1]
            names = list(transformer.get_feature_names_out())
            scale = transformer.named_transformers_["numeric"].scale_
            numeric = list(transformer.named_transformers_["numeric"].feature_names_in_)
            coefficient = ridge.coef_[names.index("numeric__days_since_start")] / scale[numeric.index("days_since_start")]
        else:
            coefficient = self.trend_model.coef_[TREND_FEATURES.index("days_since_start")]
        return float(np.exp(coefficient * 30) - 1)

    def feature_importance(self) -> pd.Series:
        gain = self.estimator.booster_.feature_importance(importance_type="gain")
        names = self.estimator.booster_.feature_name()
        return pd.Series(gain, index=names).sort_values(ascending=False) / gain.sum()


class BlendModel(RateModel):
    """Weighted average of member models in log space (a weighted geometric mean of rates)."""

    name = "Blend: LightGBM + Ridge"

    def __init__(self, context: ModelContext, weights: dict[type, float] | None = None):
        super().__init__(context)
        self.weights = weights or {LightGBMModel: 0.5, RidgeModel: 0.5}

    def fit(self, frame: pd.DataFrame) -> "BlendModel":
        self.members = [(cls(self.context).fit(frame), weight) for cls, weight in self.weights.items()]
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        total = sum(weight for _, weight in self.members)
        return sum(model.predict_log_rpm(frame) * weight for model, weight in self.members) / total

    def member(self, cls: type) -> RateModel:
        return next(model for model, _ in self.members if isinstance(model, cls))
