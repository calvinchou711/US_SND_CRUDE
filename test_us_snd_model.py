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

def test_month_alignment_complete_balance_and_spr(panel):
    assert set(panel.padd)=={1,2,3,4,5}
    assert panel.groupby('month').padd.nunique().eq(5).all()
    np.testing.assert_allclose(panel.balance_kb,panel[FLOWS].to_numpy()@SIGNS)
    np.testing.assert_allclose(panel.stock_kb,panel.commercial_stock_kb+panel.spr_stock_kb)
    assert panel[panel.padd.ne(3)].spr_stock_kb.eq(0).all()
    known = panel.dropna(subset=['previous_stock_kb'])
    np.testing.assert_allclose(known.accounting_residual_kb-known.reported_change_residual_kb,
        known.reported_stock_change_kb-known.balance_kb,atol=1e-7)
    ordinary=panel[~panel.month.eq(pd.Timestamp('2026-01-01'))]
    assert ordinary.accounting_residual_kb.abs().max()<=2
    assert panel.accounting_residual_kb.abs().max()<=60
    assert panel.five_term_residual_kb.abs().mean()>100*panel.accounting_residual_kb.abs().mean()

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

def test_legacy_candidate_matches_original_regression(panel):
    from sklearn.linear_model import LinearRegression
    f=supervised(panel[panel.padd.eq(3)].iloc[:100].copy())
    train,test=f.iloc[:-12],f.iloc[-12:]
    old=LinearRegression(positive=True).fit(train[columns('legacy_constrained_level')],train.actual_kb)
    model,_=fit('legacy_constrained_level',train)
    np.testing.assert_allclose(predict('legacy_constrained_level',model,test),np.maximum(old.predict(test[columns('legacy_constrained_level')]),0))

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
    np.testing.assert_allclose(regional.stock_change_kb,regional.balance_kb+regional.model_reconciliation_kb,atol=1e-7)

def test_aggregation_rejects_partial_and_duplicate_regions():
    frame=pd.DataFrame({'month':[1]*4,'padd':[1,2,3,4],'value':[1]*4})
    with pytest.raises(ValueError,match='all five'):aggregate(frame,['month'],['value'])
    with pytest.raises(ValueError,match='Duplicate'):aggregate(pd.concat([frame,frame]),['month'],['value'])

def test_source_integrity():
    manifest=json.loads((HERE/'data/source_manifest.json').read_text())
    assert len(manifest)==55
    for s in manifest:
        raw=(HERE/'data/raw'/f"{s['series_id']}m.xls").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==s['sha256']

def test_selection_only_uses_development_predictions():
    output=HERE/'model_output'
    metadata=json.loads((output/'model_metadata.json').read_text())
    p=pd.read_csv(output/'us_predictions.csv',parse_dates=['month'])
    cv=p.query("split=='cv' and model!='deployed_policy'").copy()
    cv['error']=abs(cv.actual_kb-cv.predicted_kb)
    assert cv.groupby('model').error.mean().idxmin()==metadata['deployed_policy']
    r=pd.read_csv(output/'us_recursive_cv_predictions.csv',parse_dates=['month','origin_month'])
    assert r.month.max()<pd.Timestamp(metadata['holdout_start'])
    assert r.origin_month.nunique()==6
    r['error']=abs(r.actual_kb-r.predicted_kb)
    assert r.groupby('model').error.mean().idxmin()==metadata['horizon_policy']
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
    regional=pd.read_csv(output/'padd_forecast_12m.csv')
    national=pd.read_csv(output/'us_forecast_12m.csv')
    assert len(regional)==60 and len(national)==12
    np.testing.assert_allclose(national.stock_kb,regional.groupby('month').stock_kb.sum())
    np.testing.assert_allclose(regional.supply_kb-regional.total_demand_kb,regional.balance_kb,atol=1e-7)
    assert len(pd.read_csv(output/'latest_forecast.csv'))==6
