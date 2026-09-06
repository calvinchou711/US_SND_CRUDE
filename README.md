# U.S. crude supply-and-demand model — reviewed September 2026

Start with **[us_snd_model_results.ipynb](us_snd_model_results.ipynb)**, an executed
74-cell walkthrough modeled on the gasoline notebook. It explains the original
implementation's issues, sources, all 16 candidates, chronological model selection,
a separate final evaluation period, JODI context, recursive tests, and forecasts.
All implementation code and review findings are included in this single notebook;
the companion review document is optional. No project Python modules are imported
when the notebook runs. Share the notebook with its `data/` folder for offline
reproduction, or read its saved tables and figures without running anything.

## What changed

The model now reconciles flows with the stock change in the **same month** and
includes net inter-PADD receipts, supply adjustments, transfers to crude supply,
and direct crude use. Forecasts use **only prior-month observations**. Historical
accounting and predictive performance are evaluated separately.

The stock target remains **total crude oil including SPR**, as in the original
model. Commercial and SPR stocks are explicitly separated in the historical data
and notebook. The model does not forecast prices or separately predict SPR policy.

Fifty-five EIA source spreadsheets were downloaded through **June 2026**. The
analysis uses January 2007 onward. Original XLS files, normalized data, retrieval
timestamps, source titles/URLs, and SHA-256 hashes are preserved under `data/`.
JODI crude history is snapshotted from the shared database, ending January 2026.

## Results and limitations

One-month model selection uses 10 expanding annual test folds (July 2014–June 2024),
then evaluates on July 2024–June 2026. All 16 candidates are compared on the same
dates. National selection compares both a common model family and the mix of
regional CV winners after summing the five PADD predictions.

The current experiment optimizes model hyperparameters with **GridSearchCV and
10-fold TimeSeriesSplit**, separately for each PADD. Its objective is stock-level
MAE, including the stock floor, damping and rolling-window training rules.
The final 24 months are excluded from every search. Development scores reuse the
tuning folds and are selection scores, not nested CV performance estimates.

Twelve-month selection uses six disjoint development years. At each origin,
parameters are searched again using ten folds entirely within earlier history.
The final-period recursive forecasts and future forecasts use parameters selected
before the final period; coefficients are refitted as observations become available.
The 13 final recursive paths overlap, so their errors are correlated.

The executed notebook and `model_output/us_model_metrics.csv` report current
performance. Chosen settings are in `model_output/best_parameters.csv`; all grid
candidates and fold scores are in `model_output/grid_search_results.csv`.
The previous fixed-setting outputs and notebook are preserved under
`model_output_before_grid_search_20260906/` for comparison.

In the September 6 run, seasonal change remains the selected national policy for
both horizons. Its one-month holdout MAE remains 11,662 kb. The tuned random forest
has the lowest one-month holdout MAE, 10,352 kb, improving 7.5% from its prior
11,195 kb; this holdout ranking does not change development-based selection.
Other families have mixed holdout results. Some neural-network grid fits reached
solver convergence limits; those warnings are retained in `fit_warnings.csv`.
All 350 searches (34,650 fold fits) completed, and all 15 tests pass.

This is a retrospective, latest-vintage evaluation. The final period is excluded
from this experiment's selection, but earlier crude notebooks used some of these
dates. EIA publication lags/revisions are not reconstructed, and no calibrated
prediction intervals are provided.

## Parameter grids

Each row is a Cartesian product. Other estimator settings remain as declared in
`estimator()`; `candidate_models.json` exports both those settings and the grids.

| Model | Search values |
|---|---|
| Ridge, seasonal/rolling/damped ridge | alpha: 0.01, 0.1, 1, 10, 100, 1000 |
| Polynomial ridge | degree: 1, 2; alpha: 1, 10, 100, 1000 |
| Spline ridge | knots: 3, 5, 7; degree: 2, 3; alpha: 1, 100, 1000 |
| Random forest | trees: 100, 200; depth: 3, 5, unlimited; minimum leaf: 5, 12 |
| XGBoost | trees: 100, 200; depth: 2, 4; learning rate: 0.01, 0.05, 0.1; L2: 1, 30 |
| Neural network | hidden layers: (16), (32), (32,16); alpha: 0.01, 1, 10 |
| Ensemble | ridge/forest/XGBoost weights: (1,1,1), (2,1,1), (1,2,1), (1,1,2) |

Constrained OLS and the four baselines have no grid. Ensemble constituent settings
are fixed while voting weights are searched. Scaling and other transformations
are fitted separately on each fold's training observations. Every search has ten
chronological test blocks; one-month development blocks are 12 months, while
searches within recursive origins use TimeSeriesSplit's default block size.

## Accounting

Stocks are month-end thousand barrels; flows are thousand barrels per calendar month:

```text
Stock[t] - Stock[t-1]
  = production[t] + imports[t] + net receipts[t] + adjustments[t]
  + transfers to crude supply[t] - refinery input[t] - exports[t]
  - direct crude use[t] + accounting residual[t]
```

Most complete-balance residuals are within 1 kb. January 2026 has source/reporting
differences of 60 kb in PADD 2 and 6 kb in PADD 4; they are retained. They also occur
in reported stock change versus changes in month-end levels.

Sparse unreported exports/receipts and transfer-series gaps are explicitly assumed
zero and flagged. The transfer series start in January 2022; earlier values represent
zero in this published-category reconstruction. Withheld/unavailable markers are not
filled. Every national aggregation requires five PADDs on the same month.

## Run and files

```bash
cd /home/calvin/commodities/oil/us_snd_crude
python -m pip install -r requirements.txt
python us_snd_model.py --cv-folds 10 --jobs 8  # offline snapshots and grid search
python us_snd_model.py --refresh-data   # refresh EIA; snapshot JODI from DuckDB
python -m pytest -q
```

Run All in the notebook executes the data preparation, model definitions, a worked
validation example, and seven visible experiment stages, then regenerates all
comparisons, forecasts, charts, and tables.
`create_notebook.py` regenerates notebook source and clears saved execution outputs;
`execute_notebook.py` executes it and streams progress. A full grid-search run
includes tuning at six recursive development origins and takes longer than the
earlier fixed-setting run. The shared database is read-only.

For presentation-only changes after a completed experiment, use
`python create_notebook.py --preserve-outputs` followed by
`python execute_notebook.py --render-saved`. This retains saved computation outputs
and regenerates reporting from the existing CSVs; it does not rerun grid search.

`model_output/` contains:

- PADD/U.S. historical balances, commercial/SPR context, and JODI comparisons.
- All one-month CV/evaluation predictions, annual train/test diagnostics, regional
  CV winners, national policy scores, and selected-model specifications.
- Recursive development and final-period predictions and horizon metrics.
- `latest_forecast.csv`: five PADDs plus U.S., using the one-month policy.
- `padd_forecast_12m.csv` and `us_forecast_12m.csv`: long-horizon policy, forecast
  flows, raw accounting paths, and explicit `model_reconciliation_kb` differences.
- `fitted_models.joblib` and `fitted_horizon_models.joblib`: separate one-month and
  long-horizon selections. Baselines have a name and no fitted estimator object.
- Coefficients for the complete constrained comparison model, candidate settings,
  fitting warnings, and run metadata. These coefficients describe the constrained
  comparison model; the deployed policy is identified separately in metadata.
- `best_parameters.csv`: development-selected parameters for every PADD/model.
- `grid_search_results.csv`: every candidate, ten individual train/test fold scores,
  ranks, timings, search stage and last training month.

One-month and long-horizon policies are selected separately and may differ on a
future rerun. Flow forecasts use daily-rate seasonality/trend over the trailing
60 months. Statistical stock changes need not equal their flow balance; the
reconciliation term exposes that difference rather than treating it as an observed
EIA adjustment. The raw accounting stock path remains unclipped.

## Original work

`legacy_results_20260905/` is a frozen copy of the pre-review code, notebooks,
README, tests, and output folders. It is not the current analysis.
`legacy_crude_model.py` retains the original implementation at the working root
for reproducibility. The two older comparison notebooks now import that module
and display a legacy notice. The current main notebook replaces the earlier
walkthrough, whose exact pre-review copy is in the archive.

The model-output filenames used by the database builder are retained, but their
columns now use explicit target-month fields such as `stock_kb` and `balance_kb`.
No shared-database rebuild was performed.
