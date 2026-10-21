"""End-to-end pipeline: clean -> validate -> train final model -> predict.

Usage:
    python run_pipeline.py              # full run, including rolling validation
    python run_pipeline.py --skip-cv    # skip rolling validation (faster)
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

from src import clean, config, eda
from src.features import category_levels, quote_regimes
from src.models import BlendModel, LaneAverageBaseline, LightGBMModel, ModelContext, RidgeModel
from src.validate import error_breakdown, evaluate_split, holdout_split, new_city_holdout, rolling_cv

FINAL_MODEL = "LightGBM + linear trend"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-cv", action="store_true", help="skip the rolling monthly validation")
    args = parser.parse_args()

    output = config.OUTPUT_DIR
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("1. Loading and cleaning data")
    train_raw, validation_raw, december_raw = clean.load_raw()
    daily_market = clean.build_daily_market(train_raw, validation_raw)
    cities = clean.build_city_table(train_raw, validation_raw)
    categories = category_levels(train_raw, validation_raw, december_raw)

    cleaner = clean.FeatureCleaner().fit(train_raw)
    train = cleaner.transform(train_raw, daily_market)
    validation = cleaner.transform(validation_raw, daily_market)
    train["label_outlier"] = clean.flag_label_outliers(train)
    reference_loads = pd.concat(
        [train.drop(columns=[config.TARGET, "label_outlier"]), validation], ignore_index=True
    )

    quality = {
        "train_rows": len(train_raw),
        "validation_rows": len(validation_raw),
        "missing_weight_train": int(train_raw["weight"].isna().sum()),
        "negative_weight_train": int((train_raw["weight"] < 0).sum()),
        "missing_market_index_train": int(train_raw["market_index"].isna().sum()),
        "missing_weight_validation": int(validation_raw["weight"].isna().sum()),
        "negative_weight_validation": int((validation_raw["weight"] < 0).sum()),
        "missing_market_index_validation": int(validation_raw["market_index"].isna().sum()),
        "label_outliers_removed": int(train["label_outlier"].sum()),
    }
    context = ModelContext(daily_market, categories, cities, reference_loads)
    factories = {
        "Baseline: lane average": lambda: LaneAverageBaseline(context),
        "Ridge regression": lambda: RidgeModel(context),
        "LightGBM + linear trend": lambda: LightGBMModel(context),
        "Blend: LightGBM + Ridge": lambda: BlendModel(context),
    }
    train_cities = set(train_raw["pickup"]) | set(train_raw["delivery"])
    validation_cities = set(validation_raw["pickup"]) | set(validation_raw["delivery"])
    new_cities = sorted(validation_cities - train_cities)
    quality["cities_only_in_validation"] = new_cities
    quality["validation_loads_touching_new_city"] = int(
        (validation_raw["pickup"].isin(new_cities) | validation_raw["delivery"].isin(new_cities)).sum()
    )
    print(json.dumps(quality, indent=2))

    print("\n2. quote_signal regime check (label-free, uses a quote-free Ridge model)")
    reference_model = RidgeModel(context).fit(train[~train["label_outlier"]])
    regimes = quote_regimes(reference_loads, reference_model.predict_log_rpm(reference_loads))
    regimes.to_csv(output / "quote_regimes_daily.csv")
    month_regime = regimes.groupby(regimes.index.to_period("M").astype(str)).agg(
        mean_correlation=("correlation", "mean"), regime=("regime", lambda r: r.mode()[0])
    ).round(3)
    print(month_regime.to_string())
    prediction_regimes = set(regimes.loc[validation["date"].unique(), "regime"])
    quality["prediction_period_quote_regimes"] = sorted(prediction_regimes)
    if prediction_regimes != {"noise"}:
        print("WARNING: quote_signal is informative on some prediction days; the models ignore it.")

    print(f"\n3. Time-based holdout: train before {config.HOLDOUT_START}, test from {config.HOLDOUT_START}")
    fit_part, holdout = holdout_split(train)
    holdout_rows, holdout_predictions = [], {}
    for name, factory in factories.items():
        _, predicted, all_rows, clean_rows = evaluate_split(factory, fit_part, holdout)
        holdout_predictions[name] = predicted
        holdout_rows.append({"model": name, "rows": "clean labels", **clean_rows})
        holdout_rows.append({"model": name, "rows": "all labels", **all_rows})
    holdout_table = pd.DataFrame(holdout_rows).round(3)
    holdout_table.to_csv(output / "holdout_metrics.csv", index=False)
    print(holdout_table.to_string(index=False))

    seen_lanes = set(zip(fit_part["pickup"], fit_part["delivery"]))
    breakdown = error_breakdown(holdout, holdout_predictions[FINAL_MODEL], seen_lanes)
    breakdown.to_csv(output / "holdout_error_breakdown.csv")
    print(f"\n{FINAL_MODEL} holdout error breakdown (clean labels):")
    print(breakdown.to_string())

    print("\n4. New-city holdout: same time split, 8 random cities removed from training entirely")
    city_rows = []
    for name, factory in factories.items():
        held, results = new_city_holdout(factory, train)
        for group, metrics in results.items():
            city_rows.append({"model": name, "rows": group, **metrics})
    city_table = pd.DataFrame(city_rows).round(3)
    city_table.to_csv(output / "new_city_holdout_metrics.csv", index=False)
    print(f"Held-out cities: {', '.join(held)}")
    print(city_table.to_string(index=False))

    if not args.skip_cv:
        print("\n5. Rolling monthly validation (expanding window)")
        rolling = []
        for name, factory in factories.items():
            table = rolling_cv(factory, train)
            table.insert(0, "model", name)
            rolling.append(table)
        rolling_table = pd.concat(rolling).round(3)
        rolling_table.to_csv(output / "rolling_cv_metrics.csv", index=False)
        print(rolling_table.to_string(index=False))
        summary = rolling_table.groupby("model")[["MAE", "RMSE", "MAPE_%", "R2"]].mean().round(3)
        summary.to_csv(output / "rolling_cv_summary.csv")
        print("\nMean across months:")
        print(summary.to_string())
        eda.model_comparison(holdout_table, rolling_table, figures / "rolling_validation.png")

    print(f"\n6. Training final model ({FINAL_MODEL}) on all clean labeled data (Jan-Oct 2025)")
    final_model = factories[FINAL_MODEL]().fit(train[~train["label_outlier"]])
    lightgbm = final_model if isinstance(final_model, LightGBMModel) else final_model.member(LightGBMModel)
    lightgbm.feature_importance().round(4).to_csv(output / "feature_importance.csv", header=["gain_share"])
    print(f"Fitted time trend: {lightgbm.trend_per_month() * 100:+.2f}% per month")

    print("\n7. Predicting validation loads and December chart inputs")
    validation_rates = pd.Series(final_model.predict(validation), index=validation["load_id"])
    template = pd.read_csv(config.TEMPLATE_PATH)
    template["predicted_rate"] = template["load_id"].map(validation_rates).round(2)
    if template["predicted_rate"].isna().any() or (template["predicted_rate"] <= 0).any():
        raise ValueError("validation predictions contain missing or non-positive values")
    template[["load_id", "predicted_rate"]].to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(template):,} predictions to {config.PREDICTIONS_PATH.name}")
    validation_months = validation["date"].dt.to_period("M").astype(str)
    print("Mean predicted rate per mile by month:",
          (validation_rates.to_numpy() / validation["distance"]).groupby(validation_months).mean().round(3).to_dict())

    december_original = pd.read_csv(config.DECEMBER_PATH)
    december = clean.complete_december_inputs(december_raw, cities, daily_market)
    december = cleaner.transform(december, daily_market)
    december_original["predicted_rate"] = np.round(final_model.predict(december), 2)
    december_original.to_csv(config.DECEMBER_PATH, index=False)
    print(f"Filled predicted_rate in {config.DECEMBER_PATH.relative_to(config.ROOT)}")

    print("\n8. Saving exploration figures")
    eda.rate_per_mile_by_distance(train, figures / "rate_per_mile_by_distance.png")
    eda.label_outliers(train, figures / "label_outliers.png")
    eda.quote_regimes(regimes, figures / "quote_regimes.png")
    eda.residual_trend(train, figures / "residual_trend.png")
    eda.december_inputs(december.assign(predicted_rate=december_original["predicted_rate"]),
                        figures / "december_inputs.png")

    summary = {
        "data_quality": quality,
        "final_model": FINAL_MODEL,
        "trend_per_month_pct": round(lightgbm.trend_per_month() * 100, 3),
        "validation_prediction_stats": template["predicted_rate"].describe().round(2).to_dict(),
        "december_prediction_range": [float(december_original["predicted_rate"].min()),
                                      float(december_original["predicted_rate"].max())],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {output.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
