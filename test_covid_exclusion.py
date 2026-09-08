import numpy as np
import pandas as pd
import pytest
from us_snd_model import (outside_covid, training_rows, validation_splits,
    recursive_origins, supervised, forecast_flows, feature_row,
    StockRegressor, estimator, fit, predict, load_panel, FLOWS)


def test_exclusion_boundaries_and_calendar_lags():
    dates = pd.date_range('2019-01-01', '2023-01-01', freq='MS')
    frame = pd.DataFrame({'month':dates})
    kept = training_rows(frame).month
    assert pd.Timestamp('2020-02-01') in kept.values
    assert pd.Timestamp('2022-04-01') in kept.values
    assert not kept.between('2020-03-01','2022-03-01').any()
    assert (~outside_covid(dates)).sum() == 13


def test_covid_values_cannot_affect_admitted_training_rows_or_flow_fit():
    history = load_panel().query('padd == 1').reset_index(drop=True)
    altered = history.copy()
    covid = ~outside_covid(history.month)
    cols = FLOWS + ['stock_kb']
    if 'utilization_ratio' in history:
        cols += ['utilization_ratio']
    altered.loc[covid, cols] += 100000
    original = training_rows(supervised(history)).reset_index(drop=True)
    changed = training_rows(supervised(altered)).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, changed)
    months = pd.date_range('2021-07-01', periods=12, freq='MS')
    pd.testing.assert_frame_equal(forecast_flows(history[history.month.le('2021-06-01')], months),
        forecast_flows(altered[altered.month.le('2021-06-01')], months))
    # Calendar lag12 for April 2022 must still refer to April 2021.
    past = history[history.month.lt('2022-04-01')]
    assert feature_row(past, pd.Timestamp('2022-04-01'))['stock_lag12'] == history.loc[history.month.eq('2021-04-01'),'stock_kb'].iloc[0]


def test_validation_dates_and_full_year_origins_exclude_covid():
    frame = pd.DataFrame({'month':pd.date_range('2008-01-01','2024-06-01',freq='MS')})
    splits = validation_splits(frame, 10, 12)
    assert len(splits) == 10
    for tr, te in splits:
        assert max(tr) < min(te)
        assert len(te) == 12
        assert outside_covid(frame.iloc[te].month).all()
        assert outside_covid(training_rows(frame.iloc[tr]).month).all()
    origins = recursive_origins(pd.Timestamp('2024-06-01'))
    assert len(origins) == 6
    for origin in origins:
        dates = pd.date_range(origin+pd.offsets.MonthBegin(), periods=12, freq='MS')
        assert outside_covid(dates).all()
        assert max(dates) <= pd.Timestamp('2024-06-01')


@pytest.mark.parametrize('name', ['seasonal_residual_change', 'huber_change'])
def test_new_model_wrapper_matches_refit(name):
    frame = supervised(load_panel().query('padd == 1').iloc[:80].reset_index(drop=True))
    train, test = frame.iloc[:-12], frame.iloc[-12:]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        wrapper = StockRegressor(name, estimator(name)).fit(train, train.actual_kb)
    model, _ = fit(name, train)
    np.testing.assert_allclose(wrapper.predict(test), predict(name, model, test))

def test_saved_refit_excludes_covid_and_keeps_every_candidate():
    import json, joblib
    from us_snd_model import HERE, MODELS
    output = HERE/'model_output'
    meta = json.loads((output/'model_metadata.json').read_text())
    assert meta['covid_exclusion']['inclusive']
    audit = pd.read_csv(output/'training_sample_audit.csv', parse_dates=['month'])
    for p, group in audit.groupby('padd'):
        admitted = group[group.training_eligible]
        assert outside_covid(admitted.month).all()
        for lag in [1,2,12]:
            assert outside_covid(admitted.month-pd.DateOffset(months=lag)).all()
        assert len(group) - len(admitted) == 25
    forecasts = pd.read_csv(output/'padd_predictions.csv', parse_dates=['month'])
    assert outside_covid(forecasts.query("split == 'cv'").month).all()
    paths = pd.read_csv(output/'recursive_cv_predictions.csv', parse_dates=['month','origin_month'])
    assert outside_covid(paths.month).all()
    assert paths.month.max() < pd.Timestamp(meta['holdout_start'])
    fitted = joblib.load(output/'all_fitted_models.joblib')
    assert set(fitted) == {(p,name) for p in range(1,6) for name in MODELS}
