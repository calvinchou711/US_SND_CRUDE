import numpy as np

from us_snd_model import (
    DEFAULT_EIA_FILE,
    DEFAULT_JODI_FILE,
    FEATURE_COLUMNS,
    aggregate_us,
    forecast_padd_snd_12m,
    fit_padd_models,
    load_eia_padd,
    load_jodi_us,
)


def test_eia_has_all_padds_and_exact_identity():
    data = load_eia_padd(DEFAULT_EIA_FILE)
    assert set(data["padd"].unique()) == {1, 2, 3, 4, 5}
    expected = (
        data["stock_t_kb"]
        + data["production_kb"]
        - data["demand_kb"]
        + data["imports_kb"]
        - data["exports_kb"]
    )
    np.testing.assert_allclose(data["identity_stock_t1_kb"], expected)
    assert data["exports_kb"].notna().all()
    assert data["exports_imputed_zero"].any()


def test_jodi_us_benchmark_has_balance_components():
    data = load_jodi_us(DEFAULT_JODI_FILE)
    required = {
        "jodi_stock_kb",
        "jodi_production_kb",
        "jodi_demand_kb",
        "jodi_imports_kb",
        "jodi_exports_kb",
    }
    assert required.issubset(data.columns)
    assert data["month"].is_unique


def test_models_use_ten_time_series_folds_and_physical_signs():
    eia = load_eia_padd(DEFAULT_EIA_FILE)
    modeled, metrics, coefficients = fit_padd_models(eia)
    assert len(metrics) == 5
    assert (metrics["cv_folds"] == 10).all()
    assert modeled.loc[modeled["cv_fold"].notna(), "cv_fold"].between(1, 10).all()
    assert modeled.groupby("padd")["cv_fold"].nunique().eq(10).all()
    assert (coefficients[FEATURE_COLUMNS] >= 0).all().all()
    assert modeled["regression_stock_t1_kb"].notna().all()
    assert modeled["cv_regression_stock_t1_kb"].notna().any()


def test_us_is_sum_of_five_padds():
    eia = load_eia_padd(DEFAULT_EIA_FILE)
    modeled, _, _ = fit_padd_models(eia)
    jodi = load_jodi_us(DEFAULT_JODI_FILE)
    us = aggregate_us(modeled, jodi)
    sums = modeled.groupby("month")["stock_t_kb"].sum(min_count=5)
    actual = us.set_index("month")["us_stock_t_kb"]
    np.testing.assert_allclose(actual.loc[sums.index], sums)
    assert (us["padd_count"] == 5).all()


def test_twelve_month_forecast_is_complete_and_additive():
    eia = load_eia_padd(DEFAULT_EIA_FILE)
    modeled, _, coefficients = fit_padd_models(eia)
    padd_forecast, us_forecast = forecast_padd_snd_12m(modeled, coefficients)
    assert len(padd_forecast) == 60
    assert len(us_forecast) == 12
    assert padd_forecast.groupby("padd")["month"].nunique().eq(12).all()
    np.testing.assert_allclose(
        us_forecast["forecast_supply_kb"],
        us_forecast["forecast_production_kb"] + us_forecast["forecast_imports_kb"],
    )
    np.testing.assert_allclose(
        us_forecast["forecast_total_demand_kb"],
        us_forecast["forecast_refinery_demand_kb"] + us_forecast["forecast_exports_kb"],
    )
    expected_identity = (
        modeled.sort_values("month").groupby("padd")["stock_t_kb"].last().sum()
        + us_forecast["forecast_balance_kb"].cumsum()
    )
    np.testing.assert_allclose(us_forecast["identity_stock_kb"], expected_identity)
    assert (us_forecast["regression_stock_kb"] >= 0).all()
