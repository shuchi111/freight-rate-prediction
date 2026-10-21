# Freight Rate Prediction

Predicts the posted rate for freight loads in November and December 2025 from loads posted January to October 2025.
The final model is LightGBM on top of a linear trend stage, validated only with time-based splits.

See `report/REPORT.md` for the validation approach, the data split and the December chart.

## Setup

Requires Python 3.11.

```bash
python -m pip install -r requirements.txt
```

## Run

Train, validate and write all predictions (takes a few minutes):

```bash
python run_pipeline.py
```

Add `--skip-cv` to skip the rolling monthly validation for a faster run.

Score the predictions and create the December chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Re-run the notebook end to end (optional; outputs are already saved in it):

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/freight_rate_model.ipynb
```

## What gets written

| Path | Contents |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 loads to predict |
| `data/december_chart_inputs.csv` | December inputs with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | December chart from `score.py` |
| `outputs/*.csv`, `outputs/summary.json` | Holdout, new-city and rolling validation metrics, feature importance, quote regimes |
| `outputs/figures/` | Figures used in the notebook and report |

## Project layout

| Path | Purpose |
|---|---|
| `src/config.py` | Paths, outlier thresholds and validation dates |
| `src/clean.py` | Loading, input cleaning and label-error flagging |
| `src/features.py` | Feature building, lane and city encoding, quote-regime check |
| `src/models.py` | Baseline, Ridge and LightGBM models |
| `src/validate.py` | Time holdout, new-city holdout and rolling validation |
| `src/eda.py` | Figures |
| `run_pipeline.py` | End-to-end training, validation and prediction |
| `score.py` | Validates the prediction files and draws the December chart |
| `notebooks/freight_rate_model.ipynb` | Step-by-step walkthrough of the same pipeline with outputs |
| `report/REPORT.md` | Validation and split write-up |
