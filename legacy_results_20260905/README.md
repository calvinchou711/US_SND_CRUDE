# U.S. PADD crude supply-and-demand model

This directory contains a monthly crude-oil stock model built from EIA and JODI
tables in `/home/calvin/commodities/data/commodities.duckdb`. It does not import
the existing `oil/snd` model or its generated data.

`us_snd_model_results.ipynb` is an executed walkthrough of data preparation,
model fitting, holdout diagnostics, PADD aggregation, JODI comparison, and the
12-month supply, demand, balance, and stock forecasts.

`model_comparison_10fold.ipynb` compares constrained linear regression, random
forests, XGBoost, and a small neural network with identical 10-fold expanding-
window validation and explicit overfitting diagnostics.

`regression_variants_10fold.ipynb` compares level, change, seasonal, polynomial,
and spline regression formulations under the same time-series validation, with
regularization and train/test-gap checks to control overfitting.

## Method

For each of PADDs 1 through 5, the accounting baseline is:

```text
Stock_t+1 = Stock_t + Production_t - Demand_t + Imports_t - Exports_t
```

All flows are monthly thousand barrels (`kb`); stocks are ending `kb`.
EIA refiner crude input is used as crude demand. EIA source keys are generated
for each PADD from `MCR{component}{PADD}1`: `FPP` production, `RIP` refinery
input, `IMP` imports, `EXP` exports, `STP` stocks, and `SCP` reported stock
change. Blank PADD export cells are retained in an `exports_imputed_zero` flag
and treated as zero when forming the additive balance.

A separate constrained linear regression is fitted for each PADD. It uses
`stock_t`, production, negative demand, imports, and negative exports as
features. Non-negative fitted coefficients retain the physical signs of the
identity while letting the historical data estimate scale and the intercept.
The final U.S. SnD is formed only after modeling by summing the five PADDs.

Validation uses 10-fold expanding-window time-series cross-validation. Every
fold trains only on observations earlier than its test observations, preventing
future leakage. The exported metrics pool the ten out-of-fold predictions and
compare them with the raw accounting identity over the same months. After
validation, each PADD regression is refitted on all history for forecasting.

The 12-month forward flow paths use linear regressions with a time trend and
monthly fixed effects, fitted separately to each PADD's last 120 months of
production, refinery demand, imports, and exports. Supply is production plus
imports; total demand/disposition is refinery input plus exports. Forecast
flows feed the fitted PADD stock models recursively, and the five paths are
then summed to form the U.S. outlook.

## EIA and JODI roles

- EIA provides the PADD-level monthly observations.
- JODI provides national U.S. crude production, refinery input, trade, stocks,
  stock change, and statistical difference. JODI has no PADD dimension, so it
  is joined to the aggregated EIA result as an explicit national benchmark.
  It is not allocated to PADDs or treated as five regional observations.

PADD stock changes will not equal the five-term identity exactly because EIA
PADD balances also reflect inter-PADD movements, transfers, and statistical
adjustments. Those omitted flows cancel only imperfectly at the U.S. boundary;
the identity error and JODI statistical difference remain visible in output.

## Run

From this directory:

```bash
python us_snd_model.py
pytest -q
```

Optional arguments are `--database`, `--output-dir`, and `--cv-folds`.

Generated files in `model_output/`:

- `padd_monthly_model.csv`: all PADD inputs, identity, fitted values, and errors
- `us_monthly_model.csv`: sum of the five PADDs plus JODI comparisons
- `jodi_us_benchmark.csv`: normalized JODI national crude history
- `padd_model_metrics.csv`: 10-fold time-series cross-validation metrics
- `padd_model_coefficients.csv`: intercepts and learned coefficients
- `latest_forecast.csv`: next-month PADD forecasts and their U.S. sum
- `padd_forecast_12m.csv`: 12-month PADD supply, demand, balance, and stocks
- `us_forecast_12m.csv`: 12-month U.S. aggregate of the five PADD forecasts
- `model_metadata.json`: sources, units, equation, and model configuration
