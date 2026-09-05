# U.S. crude supply-and-demand model — reviewed September 2026

Start with **[us_snd_model_results.ipynb](us_snd_model_results.ipynb)**, an executed
72-cell walkthrough modeled on the gasoline notebook. It explains the original
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

The selected national rule is **seasonal stock change**: latest stock plus the
mean same-month change from the preceding 60 months. Final-period national MAE is
**11,662 kb**, versus **13,182 kb** for persistence (**11.5% lower**) and **11,670 kb**
for the original five-feature constrained regression (**0.07% lower**). The simpler
rule's recent improvement over the original regression is therefore very small.

Twelve-month selection is evaluated separately using six disjoint development
years. It also selects seasonal stock change. However, across 13 overlapping
12-month final-period forecast origins, its MAE is **39,854 kb**, versus **14,931 kb**
for persistence. **The long-horizon model has not demonstrated improvement over
unchanged stocks.** The notebook shows this failure and the persistence baseline
prominently; the saved 12-month path is a conditional scenario, not a validated
reliable outlook. Historical SPR drawdowns are a major extrapolation risk.

This is a retrospective, latest-vintage evaluation. The final period is excluded
from this experiment's selection, but earlier crude notebooks used some of these
dates. EIA publication lags/revisions are not reconstructed, and no calibrated
prediction intervals are provided.

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
python us_snd_model.py                  # offline snapshots
python us_snd_model.py --refresh-data   # refresh EIA; snapshot JODI from DuckDB
python -m pytest -q
```

Run All in the notebook executes the data preparation, model definitions, a worked
validation example, and seven visible experiment stages, then regenerates all
comparisons, forecasts, charts, and tables.
`create_notebook.py` regenerates notebook source and clears saved execution outputs;
it is not needed for ordinary use. The shared database is read-only.

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
  fitting warnings, and run metadata. Coefficients are not the deployed seasonal rule.

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
