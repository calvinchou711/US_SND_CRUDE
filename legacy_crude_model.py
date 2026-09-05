"""Monthly PADD and U.S. crude-oil supply/demand model.

The model reads only the source archives under ``data/eia`` and ``data/jodi``.
EIA supplies PADD-level observations; JODI supplies an independent U.S.-level
benchmark because JODI does not publish PADD observations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "data/commodities.duckdb"
# Backward-compatible aliases for notebooks; both now point to the database.
DEFAULT_EIA_FILE = DEFAULT_DATABASE
DEFAULT_JODI_FILE = DEFAULT_DATABASE
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "model_output"

PADD_NAMES = {
    1: "East Coast",
    2: "Midwest",
    3: "Gulf Coast",
    4: "Rocky Mountain",
    5: "West Coast",
}
EIA_COMPONENT_CODES = {
    "production_kb": "FPP",
    "demand_kb": "RIP",  # crude input to refineries is crude-oil demand
    "imports_kb": "IMP",
    "exports_kb": "EXP",
    "stock_kb": "STP",
    "reported_stock_change_kb": "SCP",
}
FEATURE_COLUMNS = [
    "stock_t_kb",
    "production_kb",
    "demand_outflow_kb",
    "imports_kb",
    "exports_outflow_kb",
]
JODI_FLOW_MAP = {
    "CLOSTLV": "jodi_stock_kb",
    "INDPROD": "jodi_production_kb",
    "REFINOBS": "jodi_demand_kb",
    "TOTIMPSB": "jodi_imports_kb",
    "TOTEXPSB": "jodi_exports_kb",
    "STOCKCH": "jodi_reported_stock_change_kb",
    "STATDIFF": "jodi_statistical_difference_kb",
}


@dataclass
class ModelResults:
    padd: pd.DataFrame
    us: pd.DataFrame
    jodi: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    forecasts: pd.DataFrame
    padd_forecast_12m: pd.DataFrame
    us_forecast_12m: pd.DataFrame


def _eia_series_key(component: str, padd: int) -> str:
    """Return the legacy EIA monthly source key for a crude PADD component."""
    return f"MCR{EIA_COMPONENT_CODES[component]}{padd}1"


def load_eia_padd(path: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Load and reshape EIA monthly crude observations from DuckDB."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Commodities database not found: {path}")
    key_map = {
        _eia_series_key(component, padd): (padd, component)
        for padd in PADD_NAMES
        for component in EIA_COMPONENT_CODES
    }
    placeholders = ", ".join("?" for _ in key_map)
    with duckdb.connect(str(path), read_only=True) as connection:
        raw = connection.execute(
            f"""
            SELECT period AS DATE, series_id AS EIA_SOURCEKEY, value AS VALUE
            FROM fundamental.eia_monthly_observation
            WHERE series_id IN ({placeholders})
            ORDER BY period, series_id
            """,
            list(key_map),
        ).fetchdf()
    if raw.empty:
        raise ValueError("No required PADD crude series were found in the EIA archive")
    raw["month"] = pd.to_datetime(raw["DATE"]).dt.to_period("M").dt.to_timestamp()
    raw[["padd", "component"]] = raw["EIA_SOURCEKEY"].map(key_map).apply(pd.Series)
    raw["VALUE"] = pd.to_numeric(raw["VALUE"], errors="coerce")

    duplicate = raw.duplicated(["month", "padd", "component"], keep=False)
    if duplicate.any():
        conflicting = (
            raw.loc[duplicate]
            .groupby(["month", "padd", "component"])["VALUE"]
            .nunique(dropna=False)
        )
        if (conflicting > 1).any():
            raise ValueError("Conflicting duplicate observations in EIA source archive")
        raw = raw.drop_duplicates(["month", "padd", "component"])

    wide = (
        raw.pivot(index=["month", "padd"], columns="component", values="VALUE")
        .reset_index()
        .sort_values(["padd", "month"])
    )
    required = list(EIA_COMPONENT_CODES)
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"Missing EIA model components: {missing}")
    # Regional export cells are blank in EIA months with no reported PADD
    # exports. Preserve that provenance in a flag and use zero in the balance,
    # which is the additive convention required to form the national total.
    wide["exports_imputed_zero"] = wide["exports_kb"].isna()
    wide["exports_kb"] = wide["exports_kb"].fillna(0.0)
    wide["padd_name"] = wide["padd"].map(PADD_NAMES)
    wide["stock_t_kb"] = wide["stock_kb"]
    wide["stock_t1_actual_kb"] = wide.groupby("padd")["stock_kb"].shift(-1)
    wide["demand_outflow_kb"] = -wide["demand_kb"]
    wide["exports_outflow_kb"] = -wide["exports_kb"]
    wide["fundamental_balance_kb"] = (
        wide["production_kb"]
        - wide["demand_kb"]
        + wide["imports_kb"]
        - wide["exports_kb"]
    )
    wide["identity_stock_t1_kb"] = wide["stock_t_kb"] + wide["fundamental_balance_kb"]
    wide["identity_error_kb"] = wide["stock_t1_actual_kb"] - wide["identity_stock_t1_kb"]
    return wide.reset_index(drop=True)


def load_jodi_us(path: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Load JODI U.S. crude data from DuckDB as a national benchmark."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Commodities database not found: {path}")
    with duckdb.connect(str(path), read_only=True) as connection:
        raw = connection.execute(
            """
            SELECT ref_area AS REF_AREA, time_period AS TIME_PERIOD,
                   energy_product AS ENERGY_PRODUCT, flow_breakdown AS FLOW_BREAKDOWN,
                   unit_measure AS UNIT_MEASURE, obs_value AS OBS_VALUE,
                   assessment_code AS ASSESSMENT_CODE
            FROM fundamental.jodi_observation
            WHERE ref_area = 'US' AND energy_product = 'CRUDEOIL'
              AND flow_breakdown IN ('CLOSTLV','INDPROD','REFINOBS','TOTIMPSB','TOTEXPSB','STOCKCH','STATDIFF')
              AND unit_measure = 'KBBL'
            """
        ).fetchdf()
    if raw.empty:
        raise ValueError("No U.S. crude observations were found in the JODI archive")
    raw["month"] = pd.to_datetime(raw["TIME_PERIOD"]).dt.to_period("M").dt.to_timestamp()
    raw["component"] = raw["FLOW_BREAKDOWN"].map(JODI_FLOW_MAP)
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    # Annual archive refreshes can leave identical revision rows. Prefer the
    # lowest JODI assessment code, then collapse exact duplicates deterministically.
    raw = raw.sort_values("ASSESSMENT_CODE").drop_duplicates(
        ["month", "component", "OBS_VALUE"], keep="first"
    )
    conflicts = raw.groupby(["month", "component"])["OBS_VALUE"].nunique(dropna=True)
    if (conflicts > 1).any():
        raw = raw.sort_values("ASSESSMENT_CODE").drop_duplicates(
            ["month", "component"], keep="first"
        )
    jodi = raw.pivot(index="month", columns="component", values="OBS_VALUE").reset_index()
    for column in JODI_FLOW_MAP.values():
        if column not in jodi:
            jodi[column] = np.nan
    jodi["jodi_fundamental_balance_kb"] = (
        jodi["jodi_production_kb"]
        - jodi["jodi_demand_kb"]
        + jodi["jodi_imports_kb"]
        - jodi["jodi_exports_kb"]
    )
    jodi["jodi_identity_stock_t1_kb"] = (
        jodi["jodi_stock_kb"] + jodi["jodi_fundamental_balance_kb"]
    )
    return jodi.sort_values("month").reset_index(drop=True)


def _fit_one_padd(frame: pd.DataFrame, cv_folds: int) -> tuple[pd.DataFrame, dict, dict]:
    complete = frame.dropna(subset=FEATURE_COLUMNS + ["stock_t1_actual_kb"]).copy()
    if len(complete) < 36:
        raise ValueError(f"PADD {frame['padd'].iloc[0]} has fewer than 36 complete observations")

    # Signed outflow features plus positive coefficients preserve the physical
    # directions in Stock_t+1 = Stock_t + Production - Demand + Imports - Exports.
    splitter = TimeSeriesSplit(n_splits=cv_folds)
    cv_prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    cv_fold = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for fold, (train_positions, test_positions) in enumerate(splitter.split(complete), start=1):
        train = complete.iloc[train_positions]
        test = complete.iloc[test_positions]
        fold_model = LinearRegression(positive=True)
        fold_model.fit(train[FEATURE_COLUMNS], train["stock_t1_actual_kb"])
        cv_prediction.loc[test.index] = fold_model.predict(test[FEATURE_COLUMNS])
        cv_fold.loc[test.index] = fold

    # Refit on all history after scoring. Only this final model is used for the
    # current one-step forecast and exported full-history fitted values.
    model = LinearRegression(positive=True)
    model.fit(complete[FEATURE_COLUMNS], complete["stock_t1_actual_kb"])
    result = frame.copy()
    valid_x = result[FEATURE_COLUMNS].notna().all(axis=1)
    result.loc[valid_x, "regression_stock_t1_kb"] = model.predict(
        result.loc[valid_x, FEATURE_COLUMNS]
    )
    result["regression_stock_change_kb"] = result["regression_stock_t1_kb"] - result["stock_t_kb"]
    result["regression_error_kb"] = result["stock_t1_actual_kb"] - result["regression_stock_t1_kb"]
    result["cv_regression_stock_t1_kb"] = cv_prediction
    result["cv_regression_error_kb"] = result["stock_t1_actual_kb"] - cv_prediction
    result["cv_fold"] = cv_fold
    result["sample"] = "initial_train"
    result.loc[result["cv_fold"].notna(), "sample"] = "cv_test"
    result.loc[result["stock_t1_actual_kb"].isna(), "sample"] = "forecast"

    scored = result.dropna(subset=["cv_regression_stock_t1_kb"])
    metrics = {
        "padd": int(frame["padd"].iloc[0]),
        "padd_name": frame["padd_name"].iloc[0],
        "cv_folds": cv_folds,
        "fit_observations": len(complete),
        "cv_observations": len(scored),
        "cv_start": scored["month"].min(),
        "cv_end": scored["month"].max(),
        "cv_mae_kb": mean_absolute_error(
            scored["stock_t1_actual_kb"], scored["cv_regression_stock_t1_kb"]
        ),
        "cv_rmse_kb": mean_squared_error(
            scored["stock_t1_actual_kb"], scored["cv_regression_stock_t1_kb"]
        ) ** 0.5,
        "cv_r2": r2_score(scored["stock_t1_actual_kb"], scored["cv_regression_stock_t1_kb"]),
        "identity_cv_mae_kb": mean_absolute_error(
            scored["stock_t1_actual_kb"], scored["identity_stock_t1_kb"]
        ),
    }
    coefficients = {"padd": metrics["padd"], "intercept_kb": model.intercept_}
    coefficients.update(dict(zip(FEATURE_COLUMNS, model.coef_)))
    return result, metrics, coefficients


def fit_padd_models(eia: pd.DataFrame, cv_folds: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one constrained regression per PADD with expanding-window CV."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    modeled, metrics, coefficients = [], [], []
    for padd in sorted(PADD_NAMES):
        fitted, score, coef = _fit_one_padd(eia.loc[eia["padd"] == padd], cv_folds)
        modeled.append(fitted)
        metrics.append(score)
        coefficients.append(coef)
    return (
        pd.concat(modeled, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(coefficients),
    )


def aggregate_us(padd: pd.DataFrame, jodi: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the five PADD balances and then join JODI national benchmarks."""
    additive = [
        "stock_t_kb",
        "stock_t1_actual_kb",
        "production_kb",
        "demand_kb",
        "imports_kb",
        "exports_kb",
        "reported_stock_change_kb",
        "fundamental_balance_kb",
        "identity_stock_t1_kb",
        "identity_error_kb",
        "regression_stock_t1_kb",
        "regression_stock_change_kb",
        "regression_error_kb",
        "cv_regression_stock_t1_kb",
        "cv_regression_error_kb",
    ]
    us = padd.groupby("month", as_index=False)[additive].sum(min_count=5)
    us = us.rename(columns={column: f"us_{column}" for column in additive})
    us["padd_count"] = padd.groupby("month")["padd"].nunique().to_numpy()
    us = us.merge(jodi, on="month", how="left", validate="one_to_one")
    us["eia_minus_jodi_stock_kb"] = us["us_stock_t_kb"] - us["jodi_stock_kb"]
    for component in ["production", "demand", "imports", "exports"]:
        us[f"eia_minus_jodi_{component}_kb"] = (
            us[f"us_{component}_kb"] - us[f"jodi_{component}_kb"]
        )
    return us.sort_values("month").reset_index(drop=True)


def latest_forecasts(padd: pd.DataFrame) -> pd.DataFrame:
    """Return each PADD's newest next-month prediction and the summed U.S. result."""
    latest = padd.sort_values("month").groupby("padd", as_index=False).tail(1).copy()
    latest["forecast_month"] = latest["month"] + pd.offsets.MonthBegin(1)
    columns = [
        "padd",
        "padd_name",
        "month",
        "forecast_month",
        "stock_t_kb",
        "fundamental_balance_kb",
        "identity_stock_t1_kb",
        "regression_stock_t1_kb",
        "regression_stock_change_kb",
    ]
    forecasts = latest[columns]
    national = {
        "padd": 0,
        "padd_name": "United States (sum of PADDs)",
        "month": latest["month"].max(),
        "forecast_month": latest["forecast_month"].max(),
    }
    for column in columns[4:]:
        national[column] = latest[column].sum(min_count=5)
    return pd.concat([forecasts, pd.DataFrame([national])], ignore_index=True)


def _seasonal_trend_design(months: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Linear time trend and fixed monthly-season effects for flow forecasts."""
    dates = pd.DatetimeIndex(pd.to_datetime(months))
    trend = ((dates.year - origin.year) * 12 + dates.month - origin.month).to_numpy()
    month_dummies = np.column_stack([(dates.month == month).astype(float) for month in range(2, 13)])
    return np.column_stack([trend, month_dummies])


def forecast_padd_snd_12m(
    padd: pd.DataFrame,
    coefficients: pd.DataFrame,
    horizon: int = 12,
    lookback_months: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast PADD flows, balances, and stocks, then aggregate to the U.S.

    Each flow uses a linear trend plus monthly fixed effects. Stock paths feed
    those forecasts recursively through the final full-history PADD stock model.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if lookback_months < 36:
        raise ValueError("lookback_months must be at least 36")

    flow_columns = ["production_kb", "demand_kb", "imports_kb", "exports_kb"]
    forecasts = []
    for padd_number in sorted(PADD_NAMES):
        history = padd.loc[padd["padd"] == padd_number].sort_values("month")
        latest_month = history["month"].max()
        future_months = pd.date_range(
            latest_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        future = pd.DataFrame({"month": future_months})
        future["padd"] = padd_number
        future["padd_name"] = PADD_NAMES[padd_number]

        for column in flow_columns:
            train = history[["month", column]].dropna().tail(lookback_months)
            origin = train["month"].iloc[0]
            flow_model = LinearRegression()
            flow_model.fit(
                _seasonal_trend_design(train["month"], origin), train[column]
            )
            future[f"forecast_{column}"] = np.maximum(
                flow_model.predict(_seasonal_trend_design(future_months, origin)), 0.0
            )

        future["forecast_supply_kb"] = (
            future["forecast_production_kb"] + future["forecast_imports_kb"]
        )
        future["forecast_refinery_demand_kb"] = future["forecast_demand_kb"]
        future["forecast_total_demand_kb"] = (
            future["forecast_demand_kb"] + future["forecast_exports_kb"]
        )
        future["forecast_balance_kb"] = (
            future["forecast_supply_kb"] - future["forecast_total_demand_kb"]
        )

        coef = coefficients.loc[coefficients["padd"] == padd_number].iloc[0]
        regression_stock = float(history["stock_t_kb"].iloc[-1])
        identity_stock = regression_stock
        regression_path, identity_path = [], []
        for row in future.itertuples(index=False):
            identity_stock = identity_stock + row.forecast_balance_kb
            signed_features = np.array(
                [
                    regression_stock,
                    row.forecast_production_kb,
                    -row.forecast_demand_kb,
                    row.forecast_imports_kb,
                    -row.forecast_exports_kb,
                ]
            )
            regression_stock = max(
                float(coef["intercept_kb"] + np.dot(coef[FEATURE_COLUMNS], signed_features)),
                0.0,
            )
            identity_path.append(identity_stock)
            regression_path.append(regression_stock)
        future["identity_stock_kb"] = identity_path
        future["regression_stock_kb"] = regression_path
        forecasts.append(future)

    padd_forecast = pd.concat(forecasts, ignore_index=True)
    additive = [
        "forecast_production_kb",
        "forecast_imports_kb",
        "forecast_supply_kb",
        "forecast_demand_kb",
        "forecast_refinery_demand_kb",
        "forecast_exports_kb",
        "forecast_total_demand_kb",
        "forecast_balance_kb",
        "identity_stock_kb",
        "regression_stock_kb",
    ]
    us_forecast = padd_forecast.groupby("month", as_index=False)[additive].sum(min_count=5)
    us_forecast.insert(1, "padd_count", 5)
    return padd_forecast, us_forecast


def build_model(
    database_path: Path = DEFAULT_DATABASE,
    cv_folds: int = 10,
) -> ModelResults:
    eia = load_eia_padd(database_path)
    jodi = load_jodi_us(database_path)
    padd, metrics, coefficients = fit_padd_models(eia, cv_folds=cv_folds)
    us = aggregate_us(padd, jodi)
    padd_forecast_12m, us_forecast_12m = forecast_padd_snd_12m(padd, coefficients)
    return ModelResults(
        padd=padd,
        us=us,
        jodi=jodi,
        metrics=metrics,
        coefficients=coefficients,
        forecasts=latest_forecasts(padd),
        padd_forecast_12m=padd_forecast_12m,
        us_forecast_12m=us_forecast_12m,
    )


def write_outputs(results: ModelResults, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.padd.to_csv(output_dir / "padd_monthly_model.csv", index=False)
    results.us.to_csv(output_dir / "us_monthly_model.csv", index=False)
    results.jodi.to_csv(output_dir / "jodi_us_benchmark.csv", index=False)
    results.metrics.to_csv(output_dir / "padd_model_metrics.csv", index=False)
    results.coefficients.to_csv(output_dir / "padd_model_coefficients.csv", index=False)
    results.forecasts.to_csv(output_dir / "latest_forecast.csv", index=False)
    results.padd_forecast_12m.to_csv(output_dir / "padd_forecast_12m.csv", index=False)
    results.us_forecast_12m.to_csv(output_dir / "us_forecast_12m.csv", index=False)
    metadata = {
        "equation": "Stock_t+1 = Stock_t + Production_t - Demand_t + Import_t - Export_t",
        "frequency": "monthly",
        "flow_unit": "thousand barrels per month",
        "stock_unit": "thousand barrels",
        "eia_source": f"{DEFAULT_DATABASE}:fundamental.eia_monthly_observation",
        "jodi_source": f"{DEFAULT_DATABASE}:fundamental.jodi_observation",
        "padds": PADD_NAMES,
        "regression": "sklearn LinearRegression(positive=True) on signed balance features",
        "validation": "10-fold expanding-window TimeSeriesSplit; final forecast model refit on all history",
        "us_method": "sum of five independently modeled PADD balances",
        "missing_value_policy": "blank EIA PADD export cells are flagged and treated as zero",
        "latest_eia_month": str(results.padd["month"].max().date()),
        "latest_jodi_month": str(results.jodi["month"].max().date()),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-folds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_model(args.database, args.cv_folds)
    write_outputs(results, args.output_dir)
    print(results.forecasts.to_string(index=False))
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
