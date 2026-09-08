"""Monthly commercial crude-oil SnD: PADD models, chronological evaluation, U.S. sums."""
from pathlib import Path
import argparse
import json
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from crude_data import HERE, PADD_NAMES, FLOWS, SIGNS, load_panel, refresh_data

BASELINES = ['persistence', 'seasonal_naive', 'seasonal_change', 'forecast_flow_identity']
LEARNED = ['constrained_level', 'ridge_change', 'seasonal_ridge_change',
           'polynomial_ridge_change', 'spline_ridge_change', 'random_forest_change',
           'xgboost_change', 'neural_network_change', 'rolling_ridge_change',
           'damped_ridge_change', 'ensemble_change']
LEARNED += ['seasonal_residual_change', 'huber_change']
MODELS = BASELINES + LEARNED
BASE_FEATURES = ['stock_lag1', 'stock_lag12', 'change_lag1'] + ['lag_' + c for c in FLOWS]
FEATURES = BASE_FEATURES + ['sin_month', 'cos_month']


COVID_START = pd.Timestamp('2020-03-01')
COVID_END = pd.Timestamp('2021-03-01')


def outside_covid(months):
    dates = pd.DatetimeIndex(months)
    return ~((dates >= COVID_START) & (dates <= COVID_END))


def training_rows(frame):
    """Exclude COVID targets and any training row whose direct inputs touch COVID."""
    dates = pd.DatetimeIndex(frame.month)
    keep = outside_covid(dates)
    # lag1 stocks/flows, lag2 for the latest change, lag12 for annual stocks.
    for lag in [1, 2, 12]:
        keep &= outside_covid(dates - pd.DateOffset(months=lag))
    return frame.loc[keep]


def validation_splits(frame, folds=10, test_size=None):
    """Split eligible target dates, retaining real calendar positions and lag dates."""
    allowed = np.flatnonzero(outside_covid(frame.month))
    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size)
    return [(allowed[tr], allowed[te]) for tr, te in splitter.split(allowed)]


def recursive_origins(development_end, count=6):
    """Use six full test years that do not overlap the excluded interval."""
    origins = []
    cutoff = development_end - pd.DateOffset(years=1)
    while len(origins) < count:
        months = pd.date_range(cutoff + pd.offsets.MonthBegin(), periods=12, freq='MS')
        if outside_covid(months).all() and outside_covid([cutoff]).all():
            origins.append(cutoff)
        cutoff -= pd.DateOffset(years=1)
    return sorted(origins)


def parameter_grid(name):
    """Conventional, compact Cartesian grids; structural baselines stay fixed."""
    if name == 'seasonal_residual_change':
        return {'ridge__alpha': [0.1, 1, 10, 100, 1000]}
    if name == 'huber_change':
        return {'regressor__huberregressor__epsilon': [1.35, 1.75],
                'regressor__huberregressor__alpha': [0.01, 1, 10]}
    if name in ['ridge_change', 'seasonal_ridge_change', 'rolling_ridge_change', 'damped_ridge_change']:
        return {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
    if name == 'polynomial_ridge_change':
        return {'polynomialfeatures__degree': [1, 2], 'ridge__alpha': [1, 10, 100, 1000]}
    if name == 'spline_ridge_change':
        return {'splinetransformer__n_knots': [3, 5, 7],
                'splinetransformer__degree': [2, 3], 'ridge__alpha': [1, 100, 1000]}
    if name == 'random_forest_change':
        return {'n_estimators': [100, 200], 'max_depth': [3, 5, None],
                'min_samples_leaf': [5, 12]}
    if name == 'xgboost_change':
        return {'n_estimators': [100, 200], 'max_depth': [2, 4],
                'learning_rate': [0.01, 0.05, 0.1], 'reg_lambda': [1, 30]}
    if name == 'neural_network_change':
        return {'regressor__mlpregressor__hidden_layer_sizes': [(16,), (32,), (32, 16)],
                'regressor__mlpregressor__alpha': [0.01, 1, 10]}
    if name == 'ensemble_change':
        return {'weights': [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]}
    return {}


class StockRegressor(RegressorMixin, BaseEstimator):
    """Score the actual stock forecast, including damping, floor and rolling fit."""
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, X, y):
        from sklearn.base import clone
        eligible = training_rows(X)
        train = eligible.tail(60) if self.name == 'rolling_ridge_change' else eligible
        target = train.actual_kb if self.name == 'constrained_level' else train.delta_kb
        if self.name == 'seasonal_residual_change':
            target = target - train.seasonal_change
        self.model_ = clone(self.model).fit(train[columns(self.name)], target)
        return self

    def predict(self, X):
        return predict(self.name, self.model_, X)


def tune(name, train, folds=10, jobs=8, splits=None):
    """Search only the supplied past history; retain every candidate and fold score."""
    grid = parameter_grid(name)
    if not grid:
        return {}, pd.DataFrame(), ''
    cv = splits if splits is not None else validation_splits(train, folds)
    search = GridSearchCV(StockRegressor(name, estimator(name)),
        {'model__' + key: value for key, value in grid.items()},
        scoring='neg_mean_absolute_error', cv=cv, n_jobs=jobs,
        refit=False, error_score='raise', return_train_score=True)
    # Threads keep warnings in this process; native numerical threads are capped.
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1), parallel_backend('threading'):
        warnings.simplefilter('always', ConvergenceWarning)
        search.fit(train, train.actual_kb)
    results = pd.DataFrame(search.cv_results_)
    results['params'] = results.params.map(lambda p: json.dumps({k.removeprefix('model__'): v for k,v in p.items()}, sort_keys=True))
    results['mean_test_mae_kb'] = -results.mean_test_score
    params = {k.removeprefix('model__'): v for k,v in search.best_params_.items()}
    warning = '; '.join(sorted({str(w.message) for w in caught}))
    return params, results, warning


def seasonal_design(months, origin):
    dates = pd.DatetimeIndex(months)
    trend = ((dates.year - origin.year) * 12 + dates.month - origin.month).to_numpy() / 12
    return np.column_stack([trend] + [(dates.month == m).astype(float) for m in range(2, 13)])


def forecast_flows(history, months):
    """Forecast daily flows and crude-input/capacity ratio from origin history."""
    train = history.loc[outside_covid(history.month)].tail(60)
    targets = [c for c in FLOWS if c != 'demand_kbd'] + ['utilization_ratio']
    model = LinearRegression().fit(
        seasonal_design(train.month, train.month.iloc[0]), train[targets])
    result = pd.DataFrame(model.predict(seasonal_design(months, train.month.iloc[0])),
                          columns=targets, index=pd.DatetimeIndex(months))
    result['utilization_ratio'] = result.utilization_ratio.clip(0, 1)
    result['cdu_capacity_kbd'] = float(history.cdu_capacity_kbd.iloc[-1])
    result['demand_kbd'] = result.utilization_ratio * result.cdu_capacity_kbd
    for c in ['production_kbd', 'imports_kbd', 'exports_kbd', 'direct_use_kbd']:
        result[c] = result[c].clip(lower=0)
    return result[FLOWS + ['cdu_capacity_kbd', 'utilization_ratio']]


def feature_row(history, month):
    last = history.iloc[-1]
    row = {'stock_lag1': last.stock_kb, 'stock_lag12': history.iloc[-12].stock_kb,
           'change_lag1': last.stock_kb - history.iloc[-2].stock_kb,
           'sin_month': np.sin(2*np.pi*month.month/12), 'cos_month': np.cos(2*np.pi*month.month/12)}
    changes = history.set_index('month').stock_kb.diff()
    clean = outside_covid(changes.index) & outside_covid(changes.index - pd.DateOffset(months=1))
    changes = changes.loc[clean].tail(60)
    seasonal = changes[changes.index.month == month.month].dropna()
    row['seasonal_change'] = float(seasonal.mean()) if len(seasonal) else 0.0
    normal_history = history.iloc[:-1].loc[lambda f: outside_covid(f.month)].tail(60)
    normal = normal_history[normal_history.month.dt.month.eq(last.month.month)].stock_kb
    row['stock_seasonal_gap'] = float(last.stock_kb - normal.mean()) if len(normal) else 0.0
    row.update({'lag_' + c: last[c] * sign for c, sign in zip(FLOWS, SIGNS)})
    return row


def supervised(history):
    rows = []
    for i in range(12, len(history)):
        past, current = history.iloc[:i], history.iloc[i]
        row = feature_row(past, current.month)
        flow = forecast_flows(past, [current.month]).iloc[0]
        row.update(month=current.month, origin_month=past.month.iloc[-1],
                   actual_kb=current.stock_kb, delta_kb=current.stock_kb-past.stock_kb.iloc[-1],
                   forecast_flow_identity=max(0.0, past.stock_kb.iloc[-1] + flow[FLOWS].to_numpy() @ SIGNS * current.month.days_in_month))
        rows.append(row)
    return pd.DataFrame(rows)


def columns(name):
    if name == 'seasonal_residual_change':
        return FEATURES + ['stock_seasonal_gap', 'seasonal_change']
    if name == 'constrained_level':
        return ['stock_lag1'] + ['lag_' + c for c in FLOWS]
    if name == 'ridge_change':
        return BASE_FEATURES
    return FEATURES


def estimator(name):
    if name == 'seasonal_residual_change':
        return make_pipeline(StandardScaler(), Ridge(alpha=100))
    if name == 'huber_change':
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            HuberRegressor(max_iter=1000)), transformer=StandardScaler())
    if name == 'constrained_level':
        return LinearRegression(positive=True)
    if name in ['ridge_change', 'seasonal_ridge_change', 'rolling_ridge_change', 'damped_ridge_change']:
        return make_pipeline(StandardScaler(), Ridge(alpha=100))
    if name == 'polynomial_ridge_change':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=1000))
    if name == 'spline_ridge_change':
        return make_pipeline(SplineTransformer(n_knots=4, degree=2, extrapolation='linear'),
                             StandardScaler(), Ridge(alpha=100))
    if name == 'random_forest_change':
        return RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=12,
                                     max_features=0.8, n_jobs=1, random_state=42)
    if name == 'xgboost_change':
        return XGBRegressor(n_estimators=120, max_depth=2, learning_rate=0.03,
            min_child_weight=12, reg_lambda=30, subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, random_state=42, objective='reg:squarederror')
    if name == 'neural_network_change':
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), alpha=10, solver='lbfgs',
                         max_iter=3000, random_state=42)), transformer=StandardScaler())
    if name == 'ensemble_change':
        return VotingRegressor([(n, estimator(n)) for n in
            ['seasonal_ridge_change', 'random_forest_change', 'xgboost_change']])
    raise ValueError(name)


def fit(name, train, params=None):
    if name in BASELINES:
        return None, ''
    train = training_rows(train)
    model = estimator(name)
    if params:
        model.set_params(**params)
    if name == 'rolling_ridge_change':
        train = train.tail(60)
    target = 'actual_kb' if name == 'constrained_level' else 'delta_kb'
    # Match the search's numerical settings, including neural-network refits.
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        values = train[target] - train.seasonal_change if name == 'seasonal_residual_change' else train[target]
        model.fit(train[columns(name)], values)
    return model, '; '.join(str(w.message) for w in caught)


def predict(name, model, frame):
    if name == 'persistence':
        return frame.stock_lag1.to_numpy()
    if name == 'seasonal_naive':
        return frame.stock_lag12.to_numpy()
    if name == 'seasonal_change':
        return np.maximum(0, frame.stock_lag1.to_numpy() + frame.seasonal_change.to_numpy())
    if name == 'forecast_flow_identity':
        return frame.forecast_flow_identity.to_numpy()
    prediction = model.predict(frame[columns(name)])
    if name == 'seasonal_residual_change':
        prediction += frame.seasonal_change.to_numpy()
    if name == 'damped_ridge_change':
        prediction *= 0.5
    if name != 'constrained_level':
        prediction += frame.stock_lag1.to_numpy()
    return np.maximum(prediction, 0)


def score(actual, predicted):
    return {'mae_kb': mean_absolute_error(actual, predicted),
            'rmse_kb': mean_squared_error(actual, predicted)**0.5,
            'r2': r2_score(actual, predicted), 'n': len(actual)}


def evaluate(samples, folds=10, holdout=24, jobs=8):
    """Choose models only on development CV; score untouched final 24 months."""
    predictions, fold_metrics, selections, fitted, warning_rows = [], [], [], {}, []
    best_parameters, searches = {}, []
    for padd, frame in samples.items():
        development = frame.iloc[:-holdout]
        if len(development) - folds*12 < 36:
            raise ValueError('Need >=36 initial training months plus 10 annual folds and holdout')
        splits = validation_splits(development, folds, test_size=12)
        for name in MODELS:
            params, results, warning = tune(name, development, folds, jobs, splits)
            best_parameters[padd, name] = params
            if not results.empty:
                searches.append(results.assign(padd=padd, model=name, stage='development',
                    train_end=development.month.max()))
                print(f'PADD {padd} {name}: grid MAE {results.mean_test_mae_kb.min():,.1f}; {params}', flush=True)
            if warning:
                warning_rows.append(dict(padd=padd, model=name, stage='grid_search', warning=warning))
            for fold, (tr, te) in enumerate(splits, 1):
                train, test = training_rows(development.iloc[tr]), development.iloc[te]
                model, warning = fit(name, train, params)
                pred = predict(name, model, test)
                fold_metrics.append(dict(padd=padd, model=name, fold=fold,
                    train_end=train.month.max(), test_start=test.month.min(), test_end=test.month.max(),
                    train_mae_kb=mean_absolute_error(train.actual_kb, predict(name, model, train)),
                    **score(test.actual_kb, pred)))
                output = test[['month', 'origin_month', 'actual_kb']].copy()
                output['predicted_kb'], output['padd'], output['model'] = pred, padd, name
                output['split'], output['fold'] = 'cv', fold
                predictions.append(output)
                if warning:
                    warning_rows.append(dict(padd=padd, model=name, stage=f'cv_{fold}', warning=warning))
        current = pd.concat(predictions, ignore_index=True)
        current = current[current.padd.eq(padd)]
        losses = current.assign(error=lambda x: abs(x.actual_kb-x.predicted_kb)).groupby('model').error.mean()
        winner = losses.idxmin()
        selections.append(dict(padd=padd, selected_model=winner, cv_mae_kb=losses[winner]))
        for name in MODELS:
            model, warning = fit(name, development, best_parameters[padd, name])
            test = frame.iloc[-holdout:]
            output = test[['month', 'origin_month', 'actual_kb']].copy()
            output['predicted_kb'] = predict(name, model, test)
            output['padd'], output['model'], output['split'], output['fold'] = padd, name, 'holdout', 0
            predictions.append(output)
            if warning:
                warning_rows.append(dict(padd=padd, model=name, stage='holdout', warning=warning))
        final, warning = fit(winner, frame, best_parameters[padd, winner])
        fitted[padd] = {'name': winner, 'estimator': final}
        if warning:
            warning_rows.append(dict(padd=padd, model=winner, stage='final', warning=warning))
        print(f'PADD {padd}: selected {winner}; CV MAE {losses[winner]:,.1f} kb', flush=True)
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_metrics),
            pd.DataFrame(selections), fitted,
            pd.DataFrame(warning_rows, columns=['padd','model','stage','warning']),
            best_parameters, pd.concat(searches, ignore_index=True))


def aggregate(frame, keys, additive):
    """Reject partial U.S. totals instead of silently summing fewer than five PADDs."""
    if frame.duplicated(keys + ['padd']).any():
        raise ValueError('Duplicate PADD rows in aggregation')
    if not frame.groupby(keys).padd.nunique().eq(5).all():
        raise ValueError('U.S. aggregation requires all five PADDs')
    return frame.groupby(keys, as_index=False)[additive].sum(min_count=5)


def forecast(history, fitted, horizon=12):
    all_paths = []
    for p, state in fitted.items():
        past = history[history.padd.eq(p)].sort_values('month').copy()
        origin = past.month.iloc[-1]
        months = pd.date_range(origin + pd.offsets.MonthBegin(), periods=horizon, freq='MS')
        flows = forecast_flows(past, months)
        identity_stock = past.stock_kb.iloc[-1]
        for h, month in enumerate(months, 1):
            previous = past.stock_kb.iloc[-1]
            flow = flows.loc[month]
            balance = flow[FLOWS].to_numpy() @ SIGNS
            volume = balance * month.days_in_month
            features = feature_row(past, month)
            features['forecast_flow_identity'] = max(0.0, previous + volume)
            stock = float(predict(state['name'], state['estimator'], pd.DataFrame([features]))[0])
            identity_stock += volume
            row = dict(month=month, origin_month=origin, horizon=h, padd=p,
                padd_name=PADD_NAMES[p], model=state['name'], stock_kb=stock,
                previous_stock_kb=previous, stock_change_kb=stock-previous,
                identity_stock_unclipped_kb=identity_stock, balance_kbd=balance,
                model_reconciliation_kbd=(stock-previous)/month.days_in_month-balance,
                supply_kbd=flow.production_kbd+flow.imports_kbd+flow.net_receipts_kbd+flow.adjustments_kbd+flow.transfers_kbd,
                total_demand_kbd=flow.demand_kbd+flow.exports_kbd+flow.direct_use_kbd, **flow.to_dict())
            all_paths.append(row)
            past = pd.concat([past, pd.DataFrame([row])], ignore_index=True)
    regional = pd.DataFrame(all_paths)
    sums = FLOWS + ['stock_kb', 'previous_stock_kb','stock_change_kb', 'identity_stock_unclipped_kb',
                   'balance_kbd','model_reconciliation_kbd','supply_kbd','total_demand_kbd','cdu_capacity_kbd']
    national = aggregate(regional, ['month','origin_month','horizon'], sums)
    national['utilization_ratio'] = national.demand_kbd / national.cdu_capacity_kbd
    return regional, national


def metric_table(predictions, keys):
    return pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
        **score(g.actual_kb, g.predicted_kb)} for k,g in predictions.groupby(keys)])


def build_model(data_dir=HERE/'data', output_dir=HERE/'model_output', folds=10, jobs=8):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = load_panel(data_dir)
    histories = {p: panel[panel.padd.eq(p)].reset_index(drop=True) for p in PADD_NAMES}
    samples = {p: supervised(history) for p,history in histories.items()}
    predictions, fold_metrics, selection, fitted, fit_warnings, best_parameters, grid_results = evaluate(samples, folds, jobs=jobs)
    selected = predictions.merge(selection[['padd','selected_model']], on='padd')
    selected = selected[selected.model.eq(selected.selected_model)].copy()
    selected['model'] = 'selected_padd_models'
    combined = pd.concat([predictions, selected[predictions.columns]], ignore_index=True)
    us_predictions = aggregate(combined, ['month','origin_month','model','split','fold'], ['actual_kb','predicted_kb'])
    # Choose the aggregation model on development predictions only. A single
    # common model family can outperform the mix of regional winners nationally.
    one_month_model_scores = metric_table(us_predictions[us_predictions.split.eq('cv')], ['model'])
    one_month_model = one_month_model_scores.sort_values(['mae_kb','model']).iloc[0].model
    regional_selection = selection.copy()
    if one_month_model != 'selected_padd_models':
        for p, frame in samples.items():
            estimator, warning = fit(one_month_model, frame, best_parameters[p, one_month_model])
            fitted[p] = {'name': one_month_model, 'estimator': estimator}
            if warning:
                fit_warnings.loc[len(fit_warnings)] = [p, one_month_model, 'national_model_final', warning]
        selection['selected_model'] = one_month_model
        selection['cv_mae_kb'] = [mean_absolute_error(g.actual_kb, g.predicted_kb)
            for p in selection.padd
            for g in [predictions.query("padd == @p and model == @one_month_model and split == 'cv'")]]
    padd_metrics = metric_table(combined, ['padd','model','split'])
    us_metrics = metric_table(us_predictions, ['model','split'])
    one_step_regional, one_step_national = forecast(panel, fitted, horizon=1)
    # Long-horizon model selection is a different task. Six disjoint development
    # years end before the final holdout; all candidates use the actual recursive
    # forecast function, without seeing any test-year observations.
    development_end = samples[1].month.iloc[-25]
    recursive_cv = []
    for cutoff in recursive_origins(development_end):
        for name in MODELS:
            states = {}
            for p in PADD_NAMES:
                train = samples[p][samples[p].month.le(cutoff)]
                params, results, tuning_warning = tune(name, train, folds, jobs)
                if not results.empty:
                    grid_results = pd.concat([grid_results, results.assign(padd=p, model=name,
                        stage='recursive_development', train_end=cutoff)], ignore_index=True)
                if tuning_warning:
                    fit_warnings.loc[len(fit_warnings)] = [p, name, 'recursive_grid_search', tuning_warning]
                model, warning = fit(name, train, params)
                states[p] = {'name': name, 'estimator': model}
                if warning:
                    fit_warnings.loc[len(fit_warnings)] = [p, name, 'recursive_cv', warning]
            path, _ = forecast(panel[panel.month.le(cutoff)], states)
            path = path[['padd','month','origin_month','horizon','stock_kb']].rename(columns={'stock_kb':'predicted_kb'})
            path = path.merge(panel[['padd','month','stock_kb']], on=['padd','month'], validate='one_to_one').rename(columns={'stock_kb':'actual_kb'})
            path['model'] = name
            recursive_cv.append(path)
        print(f'Recursive development origin {cutoff.date()} complete', flush=True)
    recursive_cv = pd.concat(recursive_cv, ignore_index=True)
    us_recursive_cv = aggregate(recursive_cv, ['month','origin_month','horizon','model'], ['actual_kb','predicted_kb'])
    year_ahead_model_scores = metric_table(us_recursive_cv, ['model']).sort_values(['mae_kb','model'])
    year_ahead_model = year_ahead_model_scores.iloc[0].model
    horizon_fitted = {}
    for p in PADD_NAMES:
        model, warning = fit(year_ahead_model, samples[p], best_parameters[p, year_ahead_model])
        horizon_fitted[p] = {'name': year_ahead_model, 'estimator': model}
        if warning:
            fit_warnings.loc[len(fit_warnings)] = [p, year_ahead_model, 'horizon_final', warning]
    regional, national = forecast(panel, horizon_fitted)
    print(f'National one-step model: {one_month_model}; 12-month model: {year_ahead_model}', flush=True)
    # Thirteen rolling 12-month paths lying wholly inside the final holdout.
    # They overlap and are correlated; do not interpret them as 13 independent years.
    recursive = []
    evaluation_models = list(dict.fromkeys([year_ahead_model, one_month_model, 'persistence', 'constrained_level']))
    for offset in range(24, 11, -1):
        cutoff = panel.month.max() - pd.DateOffset(months=offset)
        for name in evaluation_models:
            states = {}
            for p, state in fitted.items():
                train = samples[p][samples[p].month.le(cutoff)]
                model, warning = fit(name, train, best_parameters[p, name])
                states[p] = {'name': name, 'estimator': model}
                if warning:
                    fit_warnings.loc[len(fit_warnings)] = [p, name, 'recursive_holdout', warning]
            path, _ = forecast(panel[panel.month.le(cutoff)], states)
            path['model'] = name
            path = path.merge(panel[['padd','month','stock_kb']], on=['padd','month'], suffixes=('_forecast','_actual'), validate='one_to_one')
            path = path.rename(columns={'stock_kb_forecast':'predicted_kb','stock_kb_actual':'actual_kb'})
            recursive.append(path)
    recursive = pd.concat(recursive, ignore_index=True)
    us_recursive = aggregate(recursive, ['month','origin_month','horizon','model'], ['actual_kb','predicted_kb'])
    us_history = aggregate(panel, ['month'], FLOWS+['stock_kb','cdu_capacity_kbd',
        'balance_kbd','accounting_residual_kbd','five_term_balance_kbd','five_term_residual_kbd'])
    us_history['utilization_ratio'] = us_history.demand_kbd / us_history.cdu_capacity_kbd
    latest_us = one_step_national.assign(padd=0, padd_name='United States (sum of PADDs)',
                                                       model=one_month_model)
    latest = pd.concat([one_step_regional, latest_us], ignore_index=True)
    tables = {'padd_monthly_model':panel,'us_monthly_model':us_history,
        'padd_predictions':combined,'us_predictions':us_predictions,'padd_model_metrics':padd_metrics,
        'us_model_metrics':us_metrics,'fold_metrics':fold_metrics,'selected_models':selection,
        'regional_cv_winners':regional_selection, 'national_model_cv_metrics':one_month_model_scores,
        'recursive_cv_predictions':recursive_cv, 'us_recursive_cv_predictions':us_recursive_cv,
        'year_ahead_model_cv_metrics':year_ahead_model_scores,
        'padd_forecast_12m':regional,'us_forecast_12m':national,'latest_forecast':latest,
        'recursive_holdout_predictions':recursive,'us_recursive_holdout_predictions':us_recursive,
        'recursive_holdout_metrics':metric_table(recursive,['padd','model']),
        'us_recursive_horizon_metrics':metric_table(us_recursive,['horizon','model']), 'fit_warnings':fit_warnings,
        'grid_search_results':grid_results,
        'best_parameters':pd.DataFrame([{'padd':p, 'model':name, 'params':json.dumps(params, sort_keys=True)}
            for (p,name),params in best_parameters.items()])}
    tables['training_sample_audit'] = pd.concat([frame[['month','origin_month']].assign(
        padd=p, training_eligible=frame.index.isin(training_rows(frame).index),
        evaluation_eligible=outside_covid(frame.month)) for p, frame in samples.items()], ignore_index=True)
    for name, frame in tables.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    all_fitted = {}
    for p, frame in samples.items():
        for name in MODELS:
            model, warning = fit(name, frame, best_parameters[p, name])
            all_fitted[p, name] = {'name': name, 'estimator': model}
    joblib.dump(all_fitted, output/'all_fitted_models.joblib')
    joblib.dump(fitted, output/'fitted_models.joblib')
    joblib.dump(horizon_fitted, output/'fitted_horizon_models.joblib')
    coefficients = []
    for p, frame in samples.items():
        model, _ = fit('constrained_level', frame)
        coefficients.append({'padd':p,'intercept_kb':model.intercept_,**dict(zip(columns('constrained_level'),model.coef_))})
    pd.DataFrame(coefficients).to_csv(output/'constrained_model_coefficients.csv',index=False)
    pd.DataFrame(coefficients).to_csv(output/'padd_model_coefficients.csv',index=False)
    metadata = {'product':'Commercial crude oil; excludes strategic reserves',
        'covid_exclusion': {'start':'2020-03-01', 'end':'2021-03-01', 'inclusive':True,
            'method':'Exclude targets and direct lag inputs touching COVID from stock fitting; exclude COVID observations from flow fits and seasonal averages; preserve the full calendar and historical balance.',
            'evaluation':'Exclude COVID target months from development selection; full recursive development years avoid COVID; final 24 months unchanged.'},
        'stock_units':'thousand barrels at month end', 'flow_units':'thousand barrels per day',
        'equation':'S[t] = S[t-1] + days[t] * (field production[t] + imports[t] + net receipts[t] + adjustments[t] + transfers to crude[t] - refinery input[t] - exports[t] - direct use[t] + residual[t])',
        'data_start':str(panel.month.min().date()),'data_end':str(panel.month.max().date()),
        'cv':f'{folds} expanding folds, 12 eligible test months each; final 24 months excluded from selection',
        'holdout_start':str(samples[1].month.iloc[-24].date()),
        'forecast_timing':'One month after latest observed EIA month; conditional on prior monthly data being available. Current revised history, not a real-time release-vintage backtest.',
        'selection':'GridSearchCV per PADD/model on development data; PADD winners plus common-family models; lowest national development MAE chooses deployment; no holdout tuning',
        'hyperparameter_search':{'method':'GridSearchCV', 'folds':folds, 'scoring':'neg_mean_absolute_error',
            'splitter':'TimeSeriesSplit on non-COVID target dates; 12 eligible observations per development block; default block size inside recursive origins',
            'evaluation':'Development scores reuse tuning folds and are selection scores, not nested CV estimates. Final 24 months excluded from all searches.',
            'recursive':'Each development origin retunes on its past history. Holdout and final refits freeze parameters selected before holdout.'},
        'one_month_model':one_month_model,
        'year_ahead_model':year_ahead_model,
        'horizon_selection':'Lowest national MAE across six disjoint 12-month development paths, six full non-COVID development years; all 17 candidates compared',
        'flow_forecast':'Last 60 non-COVID observations, daily rates, linear trend and month fixed effects; crude run = forecast utilization ratio (clipped 0–1) times CDU capacity fixed at origin',
        'forecast_stock_floor':0, 'uncertainty':'Point forecasts only; no calibrated prediction intervals',
        'recursive_validation':'13 overlapping 12-month paths entirely in final holdout; refit at each origin; correlated errors',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'sklearn':sklearn.__version__, 'xgboost':xgboost.__version__},
        'sources':['https://www.eia.gov/dnav/pet/pet_sum_snd_d_r10_mbbl_m_cur.htm',
                   'https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp']}
    (output/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    settings = {name: {'features': columns(name), 'estimator': repr(estimator(name)), 'param_grid':parameter_grid(name)}
                for name in LEARNED}
    (output/'candidate_models.json').write_text(json.dumps(settings, indent=2))
    print(us_metrics[us_metrics.split.eq('holdout')].sort_values('mae_kb').to_string(index=False))
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=HERE/'data')
    parser.add_argument('--output-dir', type=Path, default=HERE/'model_output')
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data(args.data_dir)
    build_model(args.data_dir,args.output_dir,args.cv_folds,args.jobs)
