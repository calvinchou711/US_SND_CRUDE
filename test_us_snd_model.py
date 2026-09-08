import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from crude_data import HERE, load_panel, FLOWS, SIGNS
from us_snd_model import supervised, forecast, aggregate, fit, predict, columns

@pytest.fixture(scope='module')
def panel():
    return load_panel()

def test_commercial_scope_daily_conversion_and_balance(panel):
    raw = pd.read_csv(HERE/'data/eia_observations.csv', parse_dates=['month'])
    assert set(panel.padd) == {1, 2, 3, 4, 5}
    assert not any('spr' in c.lower() for c in panel)
    stocks = raw[raw.component.eq('commercial_stock_kb')]
    joined = panel.merge(stocks, on=['month', 'padd'])
    np.testing.assert_allclose(joined.stock_kb, joined.value)
    for c in FLOWS:
        source = raw[raw.component.eq(c.replace('_kbd', '_kb'))]
        joined = panel.merge(source, on=['month', 'padd'])
        np.testing.assert_allclose(joined[c] * joined.month.dt.days_in_month, joined.value)
    np.testing.assert_allclose(panel.balance_kbd, panel[FLOWS].to_numpy() @ SIGNS)
    known = panel.dropna(subset=['previous_stock_kb'])
    np.testing.assert_allclose(known.stock_change_kb,
        (known.balance_kbd + known.accounting_residual_kbd) * known.month.dt.days_in_month, atol=1e-7)


def test_utilization_forecast_fixed_capacity_and_no_direct_run_fit(panel):
    from us_snd_model import forecast_flows
    h = panel[panel.padd.eq(3)].iloc[:80].copy()
    months = pd.date_range(h.month.iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')
    f = forecast_flows(h, months)
    assert f.cdu_capacity_kbd.eq(h.cdu_capacity_kbd.iloc[-1]).all()
    assert f.utilization_ratio.between(0, 1).all()
    np.testing.assert_allclose(f.demand_kbd, f.utilization_ratio * f.cdu_capacity_kbd)
    # Direct run is excluded from fitted targets: only its observed ratio is fitted.
    h['demand_kbd'] *= 10
    pd.testing.assert_frame_equal(f, forecast_flows(h, months))
    h['cdu_capacity_kbd'] *= 2
    doubled = forecast_flows(h, months)
    np.testing.assert_allclose(doubled.utilization_ratio, f.utilization_ratio)
    np.testing.assert_allclose(doubled.demand_kbd, 2*f.demand_kbd)


def test_target_and_future_values_cannot_change_features(panel):
    history=panel[panel.padd.eq(3)].iloc[:45].copy()
    original=supervised(history)
    altered=history.copy()
    altered.loc[altered.index[-2:],FLOWS+['stock_kb']]+=100000
    changed=supervised(altered)
    features=[c for c in original if c not in ['actual_kb','delta_kb']]
    pd.testing.assert_frame_equal(original.iloc[:-1][features],changed.iloc[:-1][features])
    assert (original.origin_month<original.month).all()
    assert np.isfinite(original.seasonal_change).all()

def test_recursive_first_step_consistency_and_aggregation(panel):
    h=panel[panel.padd.eq(1)].iloc[:65].copy()
    model,_=fit('seasonal_ridge_change',supervised(h.iloc[:-1]))
    expected=predict('seasonal_ridge_change',model,supervised(h).tail(1))[0]
    frames,states=[],{}
    for p in range(1,6):
        f=h.iloc[:-1].copy();f['padd']=p;frames.append(f)
        states[p]={'name':'seasonal_ridge_change','estimator':model}
    regional,national=forecast(pd.concat(frames),states)
    np.testing.assert_allclose(regional[regional.horizon.eq(1)].stock_kb,[expected]*5)
    np.testing.assert_allclose(national.stock_kb,regional.groupby('month').stock_kb.sum())
    np.testing.assert_allclose(regional.stock_change_kb,(regional.balance_kbd+regional.model_reconciliation_kbd)*regional.month.dt.days_in_month,atol=1e-7)

def test_aggregation_rejects_partial_and_duplicate_regions():
    frame=pd.DataFrame({'month':[1]*4,'padd':[1,2,3,4],'value':[1]*4})
    with pytest.raises(ValueError,match='all five'):aggregate(frame,['month'],['value'])
    with pytest.raises(ValueError,match='Duplicate'):aggregate(pd.concat([frame,frame]),['month'],['value'])

def test_source_integrity():
    manifest=json.loads((HERE/'data/source_manifest.json').read_text())
    assert len(manifest)==50
    for s in manifest:
        raw=(HERE/'data/raw'/f"{s['series_id']}m.xls").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==s['sha256']

def test_selection_only_uses_development_predictions():
    output=HERE/'model_output'
    metadata=json.loads((output/'model_metadata.json').read_text())
    p=pd.read_csv(output/'us_predictions.csv',parse_dates=['month'])
    cv=p.query("split=='cv'").copy()
    cv['error']=abs(cv.actual_kb-cv.predicted_kb)
    assert cv.groupby('model').error.mean().idxmin()==metadata['one_month_model']
    r=pd.read_csv(output/'us_recursive_cv_predictions.csv',parse_dates=['month','origin_month'])
    assert r.month.max()<pd.Timestamp(metadata['holdout_start'])
    assert r.origin_month.nunique()==6
    r['error']=abs(r.actual_kb-r.predicted_kb)
    assert r.groupby('model').error.mean().idxmin()==metadata['year_ahead_model']
    folds=pd.read_csv(output/'fold_metrics.csv',parse_dates=['train_end','test_start','test_end'])
    assert (folds.train_end<folds.test_start).all()
    assert folds.test_end.max()<pd.Timestamp(metadata['holdout_start'])

def test_holdout_paths_and_forecast_accounting():
    output=HERE/'model_output'
    metadata=json.loads((output/'model_metadata.json').read_text())
    paths=pd.read_csv(output/'us_recursive_holdout_predictions.csv',parse_dates=['month','origin_month'])
    assert paths.origin_month.nunique()==13
    assert paths.month.min()>=pd.Timestamp(metadata['holdout_start'])
    assert (paths.month>paths.origin_month).all()
    assert paths.groupby(['origin_month','model']).size().eq(12).all()
    regional=pd.read_csv(output/'padd_forecast_12m.csv',parse_dates=['month'])
    national=pd.read_csv(output/'us_forecast_12m.csv')
    assert len(regional)==60 and len(national)==12
    np.testing.assert_allclose(national.stock_kb,regional.groupby('month').stock_kb.sum())
    np.testing.assert_allclose(regional.supply_kbd-regional.total_demand_kbd,regional.balance_kbd,atol=1e-7)
    assert regional.groupby('padd').cdu_capacity_kbd.nunique().eq(1).all()
    np.testing.assert_allclose(national.utilization_ratio, national.demand_kbd/national.cdu_capacity_kbd)
    assert len(pd.read_csv(output/'latest_forecast.csv'))==6


def test_refresh_round_trip_preserves_units_and_commercial_scope(monkeypatch, tmp_path, panel):
    from types import SimpleNamespace
    import crude_data

    def local_download(url, timeout):
        content = (HERE/'data/raw'/url.rsplit('/', 1)[-1]).read_bytes()
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(crude_data.requests, 'get', local_download)
    refreshed = crude_data.refresh_data(tmp_path)
    pd.testing.assert_frame_equal(panel, refreshed)
    manifest = json.loads((tmp_path/'source_manifest.json').read_text())
    assert len(manifest) == 50
    assert all(e['component'] != 'stock_kb' for e in manifest)


def test_saved_selected_holdout_matches_refit(panel):
    """Detect stale saved forecasts or a mismatch between reporting and model code."""
    import json
    from us_snd_model import HERE
    output = HERE/'model_output'
    selected = pd.read_csv(output/'selected_models.csv')
    params = pd.read_csv(output/'best_parameters.csv').set_index(['padd','model'])
    saved = pd.read_csv(output/'padd_predictions.csv', parse_dates=['month'])
    for row in selected.itertuples():
        history = panel[panel.padd.eq(row.padd)].reset_index(drop=True)
        samples = supervised(history)
        estimator, _ = fit(row.selected_model, samples.iloc[:-24], json.loads(params.loc[(row.padd,row.selected_model),'params']))
        expected = predict(row.selected_model, estimator, samples.iloc[-24:])
        observed = saved[(saved.padd.eq(row.padd)) & saved.model.eq(row.selected_model) & saved.split.eq('holdout')].sort_values('month')
        np.testing.assert_allclose(expected, observed.predicted_kb, rtol=1e-7, atol=1e-5)
