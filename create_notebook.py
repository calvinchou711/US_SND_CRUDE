"""Build the reviewed crude-model walkthrough. Run All to execute the experiment."""
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# U.S. crude oil supply, demand, and stocks
## Model review, chronological comparison, and forecast results

**Question:** Which monthly PADD stock model improves on the original constrained regression and on simple baselines—and does that improvement survive a separate evaluation period?

This executed walkthrough reviews the earlier crude implementation and compares **16 approaches**. Five PADD predictions are summed to form the U.S. result. Model selection uses **10 expanding annual test folds**, followed by a separate **24-month evaluation period**. Twelve-month model selection uses **six disjoint development years**, and long-horizon evaluation uses **13 overlapping forecast origins**.

**Target:** Total crude oil stocks **including the Strategic Petroleum Reserve (SPR)**, preserving the earlier model's target. Commercial stocks and the SPR are shown separately. This is an inventory model, not a crude-price model.

Read in order: review findings → data and balance → forecast timing → model candidates → development selection → final evaluation → recursive forecasts → conclusions and reproducibility.''')
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
HERE=Path.cwd()
if not (HERE/'crude_data.py').exists():
    HERE=HERE/'oil'/'us_snd_crude'
assert (HERE/'crude_data.py').exists(), 'Run from this directory or the commodities workspace root.'
sys.path.insert(0,str(HERE))
from crude_data import load_panel, FLOWS, SIGNS, PADD_NAMES
from us_snd_model import build_model, supervised, MODELS, estimator, columns
OUTPUT=HERE/'model_output'
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns',20)
pd.set_option('display.float_format',lambda x:f'{x:,.2f}')''')
md('''## 1. What the review changed

| Finding in the original work | Change in this version | Why it matters |
|---|---|---|
| Current-month flows were labeled as next month's stock identity | Reconcile flows in t with the stock change from t−1 to t | Separates correct accounting from lag-based prediction |
| Regional movements, supply adjustments, and transfers were omitted | Download and retain the complete published crude balance | Large apparent regional imbalances are explained |
| The stock target included SPR without emphasizing it | Show total, commercial, and SPR stocks separately | Policy-driven reserve changes can dominate fitted trends |
| All-history CV was the principal assessment | Reserve final 24 months from this experiment's selection | Distinguishes choosing a model from evaluating it |
| One regression was used for the outlook | Compare seasonal baselines, regularization, nonlinear models, recent history, shrinkage, and an ensemble | Improvements must beat simple alternatives |
| Regional selection need not minimize national error | Compare common-family and regional-winner policies on U.S. development MAE | Scores the actual national objective after aggregation |
| One-month validation was used to support 12-month paths | Independently test and select with recursive development forecasts | Evaluates the procedure actually used for the outlook |
| Fitting warnings were suppressed in an earlier notebook | Export all fitting warnings | Makes numerical failures visible |

Original files and results are preserved under `legacy_results_20260905/`. The two older comparison notebooks are marked legacy and import `legacy_crude_model.py`. The original five-feature regression is also **refitted and rescored on exactly the same data and dates** here as `legacy_constrained_level`.

The old regression's lagged-flow inputs were usable predictors; the key timing error was calling its current-flow arithmetic an exact next-month accounting identity. The review does not label all earlier regression results as target leakage.''')
md('''## 2. Sources, units, and stock scope

The fresh download contains **55 EIA XLS files**, eleven series per PADD. The analysis uses January 2007 onward. Source titles, URLs, timestamps, and SHA-256 hashes are in `data/source_manifest.json`, with original files in `data/raw/`.

Stocks are **thousand barrels at month end**. Flows are **thousand barrels per calendar month**. Divide kb by 1,000 to obtain the million-barrel units shown in charts.

The published crude balance includes net inter-PADD receipts, adjustments, and transfers to crude supply in addition to production, imports, refinery inputs, exports, and direct crude use. See [EIA definitions](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp) and [the PADD 3 balance](https://www.eia.gov/dnav/pet/pet_sum_snd_d_r30_mbbl_m_cur.htm).

Commercial stock series come from [EIA non-SPR stocks](https://www.eia.gov/dnav/pet/pet_stoc_typ_a_epc0_sax_mbbl_m.htm). SPR stocks are the difference between reported total and commercial stocks, not an estimated allocation. In these data the difference is entirely in PADD 3.''')
code('''panel=load_panel(HERE/'data')
manifest=pd.read_json(HERE/'data/source_manifest.json')
display(manifest[manifest.padd.eq(3)][['component','series_id','title']])
display(panel.groupby('padd').agg(start=('month','min'),end=('month','max'),months=('month','size')))
flags=[c for c in panel if c.endswith('_assumed_zero')]
display(panel.groupby('padd')[flags].sum())''')
md('''No-data-reported/absent cells in exports and net receipts are explicitly assumed zero and flagged. The transfer series begin in January 2022; earlier transfer rows are structural zeros in the published-category reconstruction. These are reporting conventions, not claims that the underlying physical processes were absent. Unavailable/withheld text markers remain missing; unresolved core data fail validation. The loader requires a common complete endpoint and monthly calendar for all five PADDs.

The review also corrected the shared downloader's Excel parsing: literal `NA` markers must remain distinct from blank cells. No explicit unavailable markers occur in the current analysis history, so this parser correction does not alter the saved gasoline results.''')
code('''national_history=panel.groupby('month')[['stock_kb','commercial_stock_kb','spr_stock_kb']].sum()
fig,axes=plt.subplots(1,2,figsize=(14,4))
(national_history/1000).rename(columns={'stock_kb':'Total crude (target)','commercial_stock_kb':'Commercial','spr_stock_kb':'SPR'}).plot(ax=axes[0])
for p,g in panel.groupby('padd'):
    axes[1].plot(g.month,g.stock_kb/1000,label=f'PADD {p}')
axes[0].set_title('The stock target includes the SPR');axes[1].set_title('Total crude by PADD');axes[1].legend()
for ax in axes:ax.set_ylabel('Million barrels')
plt.tight_layout();plt.show()''')
md(r'''## 3. Reconstruct the balance before forecasting

\[
S_t-S_{t-1}=P_t+I_t+N_t+A_t+T_t-R_t-X_t-U_t+\epsilon_t
\]

| Symbol | Meaning |
|---|---|
| S | Total crude stocks including SPR |
| P / I | Field production / imports |
| N / A / T | Net receipts / supply adjustments / transfers to crude supply |
| R / X / U | Refinery input / exports / direct crude use |
| ε | Remaining accounting/reporting difference |

The original five-term balance omitted N, A, T, and U. The revised residual is much smaller. A small historical residual validates alignment and coverage, **not forecast skill**: published balancing terms are not all independently measured information.''')
code('''audit=panel.groupby('padd').agg(
    old_balance_mae_kb=('five_term_residual_kb',lambda s:s.abs().mean()),
    complete_balance_mae_kb=('accounting_residual_kb',lambda s:s.abs().mean()),
    max_complete_residual_kb=('accounting_residual_kb',lambda s:s.abs().max()))
display(audit)
exceptions=panel[panel.accounting_residual_kb.abs()>2]
display(exceptions[['month','padd','accounting_residual_kb','reported_change_residual_kb']])
fig,axes=plt.subplots(1,2,figsize=(14,3))
for p,g in panel.groupby('padd'):
    axes[0].plot(g.month,g.five_term_residual_kb/1000,label=f'PADD {p}',alpha=.7)
    axes[1].plot(g.month,g.accounting_residual_kb,label=f'PADD {p}',alpha=.7)
axes[0].set(title='Omitted terms create large apparent imbalances',ylabel='Million barrels')
axes[1].set(title='Complete-balance residual',ylabel='Thousand barrels')
axes[0].legend(ncol=3,fontsize=8);plt.tight_layout();plt.show()''')
md('''Two exceptions remain in January 2026: 60 kb in PADD 2 and 6 kb in PADD 4. They also appear when comparing month-end stock differences with EIA's separately reported stock-change series. We retain them as source/reporting discrepancies rather than forcing the balance to close. Other residuals are at rounding scale.''')
md('''## 4. Forecast timing and what the evaluation can establish

Each row predicts ending stock in month **t**, using stock and flow observations only through **t−1**, plus the known calendar month. Features contain lagged stock levels, the latest stock change, signed lagged flows, and optional seasonality. Actual target-month flows are never predictive inputs.

The flow-identity baseline forecasts target-month flows from the previous 60 months of daily rates, using a linear trend and month fixed effects. Multiplication by the target month's day count handles February and 30/31-day months. Main production, refinery demand, imports, and exports receive a zero floor; net receipts, adjustments, and transfers retain their signs.

This is a **conditional latest-vintage historical evaluation**. EIA releases monthly data with a delay and revises it. We have not reconstructed exactly what was available on each historical release date. Earlier crude notebooks also examined some of these dates, so the final 24 months are held out of **this experiment's selection**, not a never-before-seen prospective dataset. The long-horizon protocol was added during this review after observing that one-step selection was inadequate; its selection uses development data only, but the review as a whole is retrospective.''')
code('''example=supervised(panel[panel.padd.eq(3)].reset_index(drop=True))
display(example[['origin_month','month','stock_lag1','lag_production_kb','lag_demand_kb',
                 'forecast_flow_identity','actual_kb']].tail())
assert (example.origin_month<example.month).all()''')
md('''## 5. Sixteen candidates, with fixed settings

| Model | Specification | Main control |
|---|---|---|
| Persistence | Carry the latest stock forward | No fitting |
| Seasonal naive | Same month's stock last year | No fitting |
| Seasonal change | Latest stock + mean same-month change in trailing 60 months | Five-year history; zero change if no seasonal observation |
| Forecast-flow identity | Latest stock + forecast complete flow balance | 60-month daily-rate trend/seasonality |
| Legacy constrained level | Original stock + four signed lagged flows | Nonnegative coefficients |
| Complete constrained level | Stock + all eight signed lagged flows | Nonnegative coefficients |
| Ridge change | Monthly stock change on lagged stocks/flows | Standardization; α=100 |
| Seasonal ridge change | Ridge change plus calendar sine/cosine | α=100 |
| Polynomial ridge change | Quadratic seasonal-feature interactions | Standardization; α=1,000 |
| Spline ridge change | Smooth feature effects | Four knots, degree 2; α=100 |
| Random forest change | Stock-change forest | 150 trees, depth 4, minimum leaf 12 |
| XGBoost change | Boosted stock-change trees | 120 trees, depth 2, learning rate .03, L2=30 |
| Neural network change | Stock-change network | 16 hidden units, α=10; feature/target scaling |
| Rolling ridge change | Seasonal ridge using only last 60 training rows | Tests adaptation to recent regimes |
| Damped ridge change | Half of seasonal ridge's predicted change | Shrinks toward persistence |
| Ensemble change | Equal mean of seasonal ridge, forest, and XGBoost changes | Tests model averaging |

All transformations fit on training data only, all models are deterministic, and stock predictions have a zero floor. The neural network uses L-BFGS without random early-stopping splits. Nonlinear models predict stock **changes** and add them to the latest stock, which avoids requiring trees to extrapolate the absolute stock level. Regression coefficients are predictive associations, not causal physical elasticities.''')
code('''assert len(MODELS)==16
print('Complete constrained features:',columns('constrained_level'))
print('Legacy features:',columns('legacy_constrained_level'))
print(estimator('neural_network_change'))
print(estimator('ensemble_change'))''')
md('''## 6. Rebuild the experiment

This cell fits and evaluates all candidates from the saved inputs. It does not download data or write to the shared database; allow a few minutes.

**One-month selection:** Ten expanding folds each test 12 months. Learned parameters remain fixed within a fold while prior-month observations update. Baseline seasonal rules update using only available past data. We compare the mix of regional CV winners with every common model-family policy after summing five PADDs. Lowest national development MAE selects deployment.

**Twelve-month selection:** Six disjoint development years compare complete recursive paths for all 16 candidates. Lowest national MAE across their 72 forecast months selects the long-horizon policy separately. All six years end before the final evaluation period.

CV scores used for selection are optimistic. The final 24-month results measure subsequent performance; their best-in-hindsight model does not replace the selected policy.''')
code('''tables=build_model(HERE/'data',OUTPUT)
metadata=json.loads((OUTPUT/'model_metadata.json').read_text())
print('Data through:',metadata['data_end'])
print('Final evaluation starts:',metadata['holdout_start'])
print('One-month policy:',metadata['deployed_policy'])
print('Twelve-month policy:',metadata['horizon_policy'])
print('Fitting warnings:',len(tables['fit_warnings']))''')
code('''folds=tables['fold_metrics']
schedule=folds.query("padd==1 and model=='persistence'")[['fold','train_end','test_start','test_end','n']]
display(schedule)
assert (pd.to_datetime(schedule.train_end)<pd.to_datetime(schedule.test_start)).all()
assert pd.to_datetime(schedule.test_end).max()<pd.Timestamp(metadata['holdout_start'])''')
md('''## 7. Development results and overfitting checks

MAE is average absolute error; RMSE penalizes larger misses more. Scores are in kb. National metrics are computed **after** summing same-month regional predictions, not by summing regional MAEs. High stock-level R² can follow slow SPR/stock trends and should not outweigh errors in monthly changes.''')
code('''display(tables['regional_cv_winners'])
display(tables['national_policy_cv_metrics'].sort_values('mae_kb'))
cv=tables['padd_model_metrics'].query("split=='cv' and model not in ['selected_padd_models','deployed_policy']")
display(cv.pivot(index='model',columns='padd',values='mae_kb').style.highlight_min(axis=0))
gaps=folds.groupby('model')[['train_mae_kb','mae_kb']].mean().sort_values('mae_kb')
gaps['test_minus_train_kb']=gaps.mae_kb-gaps.train_mae_kb
display(gaps)
gaps[['train_mae_kb','mae_kb']].plot.barh(figsize=(10,6),title='Mean training and annual-test errors across PADDs')
plt.xlabel('Thousand barrels');plt.show()''')
code('''fig,ax=plt.subplots(figsize=(12,4))
for name in ['persistence','seasonal_change','legacy_constrained_level','xgboost_change','rolling_ridge_change']:
    g=folds[folds.model.eq(name)].groupby('test_end').mae_kb.mean()
    ax.plot(pd.to_datetime(g.index),g.values,label=name,marker='o')
ax.set(title='Annual test performance changes across regimes',ylabel='Mean PADD MAE (kb)')
ax.legend(fontsize=8);plt.show()
display(pd.read_csv(OUTPUT/'constrained_model_coefficients.csv'))''')
md('''Train/test gaps can reflect overfitting **and** changes in market regimes. They are diagnostics, not a statistical significance test. The 60-month ridge is fitted on fewer observations, while its displayed training MAE is scored over the entire fold training history for comparability; this can penalize its deliberate focus on recent data.''')
md('''## 8. Final one-month evaluation: July 2024–June 2026 in this run

The original regression is refitted on the same development history as every other learned candidate. Thus the comparison isolates model specification and evaluation design rather than changes in sample dates.''')
code('''scores=tables['us_model_metrics'].query("split=='holdout'").sort_values('mae_kb')
display(scores)
lookup=scores.set_index('model')
selected=lookup.loc['deployed_policy','mae_kb']
naive=lookup.loc['persistence','mae_kb']
legacy=lookup.loc['legacy_constrained_level','mae_kb']
display(Markdown(f"**Selected policy:** `{metadata['deployed_policy']}`. National MAE is **{selected:,.0f} kb**, "
    f"versus **{naive:,.0f} kb** for persistence (**{100*(1-selected/naive):.1f}% lower**) "
    f"and **{legacy:,.0f} kb** for the original regression (**{100*(1-selected/legacy):.2f}% lower**). "
    "The gain over the old regression on this final period is small; the main case for the simpler rule is its development performance and transparency."))
display(tables['padd_model_metrics'].query("split=='holdout' and model in ['deployed_policy','legacy_constrained_level','persistence']"))''')
code('''hold=tables['us_predictions'].query("split=='holdout'")
actual=hold[hold.model.eq('deployed_policy')]
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
axes[0].plot(actual.month,actual.actual_kb/1000,color='black',linewidth=2,label='Actual')
for name in ['deployed_policy','legacy_constrained_level','persistence']:
    g=hold[hold.model.eq(name)]
    axes[0].plot(g.month,g.predicted_kb/1000,label=name,alpha=.8)
axes[0].set(title='Final one-month evaluation: total U.S. crude stocks',ylabel='Million barrels');axes[0].legend()
axes[1].bar(actual.month,(actual.predicted_kb-actual.actual_kb)/1000,width=20)
axes[1].axhline(0,color='black');axes[1].set(ylabel='Forecast − actual (million bbl)',title='Selected-policy errors')
plt.tight_layout();plt.show()''')
md('''## 9. JODI national context

JODI CRUDEOIL observations are snapshotted from the local database and currently end earlier than the refreshed EIA history. JODI is never a predictive input here and has no PADD dimension. It is a separate archive, not necessarily an independent measurement process; national definitions, coverage, revisions, and reporting dates can differ. We show the levels and differences rather than assuming they are interchangeable targets. See the [JODI stock guide](https://www.jodidata.org/oil/support/user-guide/data-available-in-the-jodi-oil-world-database.aspx).''')
code('''history=tables['us_monthly_model']
jodi=tables['jodi_us_benchmark']
print('Latest JODI month:',jodi.month.max().date())
fig,ax=plt.subplots(figsize=(12,4))
for col,label in [('stock_kb','EIA total including SPR'),('commercial_stock_kb','EIA commercial'),('jodi_CLOSTLV','JODI crude')]:
    ax.plot(history.month,history[col]/1000,label=label)
ax.set(title='National source context',ylabel='Million barrels');ax.legend();plt.show()
display(jodi.tail(6))''')
md('''## 10. Long-horizon development selection and final evaluation

A one-month model updates with actual prior stock each month. A 12-month path feeds its own predictions forward. These are different tasks.

The table below selects the 12-month policy from six development years. Final evaluation then uses 13 monthly origins, each forecasting 12 months entirely inside the final 24-month period. The overlapping paths are **correlated**, not 13 independent years. Each origin refits using only data available through that origin, with model names fixed from development selection. We also carry forward the one-month policy and compare persistence and the original regression.''')
code('''display(tables['horizon_policy_cv_metrics'])
cv_paths=tables['us_recursive_cv_predictions']
display(cv_paths[cv_paths.model.eq('persistence')].groupby('origin_month').agg(first_target=('month','min'),last_target=('month','max')))
assert cv_paths.month.max()<pd.Timestamp(metadata['holdout_start'])
recursive=tables['us_recursive_holdout_predictions'].copy()
recursive['abs_error_kb']=(recursive.predicted_kb-recursive.actual_kb).abs()
recursive_summary=recursive.groupby('model').abs_error_kb.agg(['mean','median','max'])
display(recursive_summary)
horizon=tables['us_recursive_horizon_metrics']
display(horizon.pivot(index='horizon',columns='model',values='mae_kb'))
horizon.pivot(index='horizon',columns='model',values='mae_kb').plot(figsize=(11,4),marker='o',title='Final recursive error by forecast horizon')
plt.ylabel('National MAE (kb)');plt.show()''')
code('''recursive_mae=recursive_summary.loc['deployed_policy','mean']
persistence_mae=recursive_summary.loc['persistence','mean']
display(Markdown(f"**Long-horizon limitation:** The development-selected `{metadata['horizon_policy']}` policy "
    f"has recursive final-period MAE of **{recursive_mae:,.0f} kb**, versus **{persistence_mae:,.0f} kb** for persistence. "
    "Historical development selection does not guarantee better forecasts after a policy or market regime changes. "
    "Do not treat the selected 12-month path as a demonstrated improvement over unchanged stocks."))
fig,axes=plt.subplots(1,2,figsize=(14,4))
origins=sorted(recursive.origin_month.unique())
for origin,ax in zip([origins[0],origins[-1]],axes):
    g=recursive[recursive.origin_month.eq(origin)]
    a=g[g.model.eq('deployed_policy')]
    ax.plot(a.month,a.actual_kb/1000,color='black',label='Actual')
    for name in ['deployed_policy','legacy_constrained_level','persistence']:
        t=g[g.model.eq(name)];ax.plot(t.month,t.predicted_kb/1000,label=name)
    ax.set(title=f'Origin {pd.Timestamp(origin):%Y-%m}',ylabel='Million barrels');ax.legend(fontsize=8)
plt.tight_layout();plt.show()''')
md('''The SPR component explains a concrete risk: a trailing seasonal rule can extrapolate historical reserve drawdowns into a period with different government policy. A separate commercial inventory model plus explicit SPR scenarios is a sensible next research direction, but it has not been validated here. No new SPR policy assumptions are inserted to make the final evaluation look better. The forecast below retains the development-selected result and shows persistence prominently.''')
md('''## 11. Twelve-month supply-and-demand outlook

The outlook starts after the last observed EIA month, **June 2026**, so it spans July 2026–June 2027. Early rows may refer to already elapsed calendar months whose observations are not in this snapshot.

The complete projected flow balance feeds a raw accounting path. The statistical stock model can disagree with it; `model_reconciliation_kb = predicted stock change − projected flow balance` exposes that disagreement. It is not an observed EIA adjustment. The raw identity path remains unclipped so impossible paths would be visible.

The one-month and 12-month policies are selected separately. `latest_forecast.csv` contains the one-month policy, while `*_forecast_12m.csv` contains the long-horizon policy. They happen to select the same rule in this run; future reruns may differ.''')
code('''outlook=tables['us_forecast_12m']
display(outlook[['month','production_kb','demand_kb','imports_kb','exports_kb','transfers_kb',
    'adjustments_kb','balance_kb','stock_kb','identity_stock_unclipped_kb','model_reconciliation_kb']])
fig,axes=plt.subplots(2,2,figsize=(14,8))
for c,label in [('supply_kb','Supply including receipts/adjustments/transfers'),('total_demand_kb','Refinery input + exports + direct use')]:
    axes[0,0].plot(outlook.month,outlook[c]/1000,label=label)
axes[0,0].set(title='U.S. monthly flow outlook',ylabel='Million barrels/month');axes[0,0].legend(fontsize=7)
axes[0,1].bar(outlook.month,outlook.balance_kb/1000,width=20)
axes[0,1].axhline(0,color='black');axes[0,1].set(title='Projected complete balance',ylabel='Million barrels/month')
recent=history.tail(24)
axes[1,0].plot(recent.month,recent.stock_kb/1000,label='Actual')
axes[1,0].plot(outlook.month,outlook.stock_kb/1000,label='Development-selected policy')
axes[1,0].plot(outlook.month,outlook.identity_stock_unclipped_kb/1000,label='Raw flow identity',linestyle='--')
axes[1,0].plot(outlook.month,np.repeat(history.stock_kb.iloc[-1]/1000,12),label='Persistence baseline',linestyle=':')
axes[1,0].set(title='Conditional stock paths: no calibrated intervals',ylabel='Million barrels');axes[1,0].legend(fontsize=7)
for p,g in tables['padd_forecast_12m'].groupby('padd'):
    axes[1,1].plot(g.month,g.stock_kb/1000,label=f'PADD {p}')
axes[1,1].set(title='Regional total-crude stock paths',ylabel='Million barrels');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.tick_params(axis='x',rotation=30)
plt.tight_layout();plt.show()''')
md('''## 12. Conclusions and reproducibility

**Implemented improvements:** Correct month alignment for accounting; a complete regional balance; explicit SPR scope; historical input snapshots; candidate models on identical dates; a final period excluded from selection; selection on national error; separate recursive development tests; visible fitting warnings and model/flow discrepancies.

**What the experiments support:** A simpler seasonal-change rule improves development and one-month final-period MAE over persistence. Its final-period gain over the original regression is very small. Larger, rolling, damped, and ensemble models do not automatically improve the selected national objective. The development-selected long-horizon rule fails to beat persistence in the final period. Better validation is valuable even when it rejects a stronger performance claim.

**Remaining limitations:** Latest-vintage rather than release-vintage data; small retrospective final sample; correlated multi-horizon errors; unmodeled SPR policy changes; reporting-category changes and flagged zero assumptions; no calibrated uncertainty intervals. Commercial-only modeling with explicit reserve scenarios and actual release-vintage evaluation are future experiments, not completed performance improvements.

From this directory:

```bash
python -m pip install -r requirements.txt
python us_snd_model.py                   # saved inputs, no network
python us_snd_model.py --refresh-data    # refresh EIA; snapshot local JODI
python -m pytest -q
```

Run All reproduces the experiment and charts. `data/` preserves inputs and provenance; `model_output/` contains all predictions, metrics, chosen policies, coefficients, forecasts, and serialized one-month and long-horizon models. `MODEL_REVIEW.md` records findings and limitations. Original work is archived under `legacy_results_20260905/`. The shared database is read-only.''')
code('''display(pd.Series(metadata,name='Run configuration'))
assert not tables['padd_predictions'].duplicated(['padd','month','model','split']).any()
assert tables['padd_forecast_12m'].groupby('month').padd.nunique().eq(5).all()
assert len(outlook)==12
assert len(tables['latest_forecast'])==6
assert not ((tables['us_recursive_cv_predictions'].month>=pd.Timestamp(metadata['holdout_start']))).any()
print('Notebook checks passed. Saved result files:')
for p in sorted(OUTPUT.iterdir()):print(p.name)''')

# Copy the implementation into ordinary notebook cells at generation time.
# The emitted notebook never imports project Python files or reads their source.
import ast
import textwrap
model_source=(HERE/'us_snd_model.py').read_text()
data_source=(HERE/'crude_data.py').read_text()
def definition(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)
def definitions(source,*names):
    return '\n\n\n'.join(definition(source,n) for n in names)
def markdown(s):return nbf.v4.new_markdown_cell(s.strip())
def python(s):return nbf.v4.new_code_cell(s.strip())
setup="""from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, io, json, platform, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, duckdb, joblib, sklearn, xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from IPython.display import display, Markdown

# Only input data are external. No local Python module is imported.
HERE=Path.cwd()
if not (HERE/'data/eia_observations.csv').exists():
    HERE=HERE/'oil'/'us_snd_crude'
assert (HERE/'data/eia_observations.csv').exists(), 'Keep the data folder beside this notebook.'
DATABASE=HERE.parents[1]/'data/commodities.duckdb'  # optional JODI refresh only
OUTPUT=HERE/'model_output'
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns',20)
pd.set_option('display.float_format',lambda x:f'{x:,.2f}')
"""
body=definition(model_source,'build_model')
body=textwrap.dedent('\n'.join(body.splitlines()[1:]))
body=body[:body.rfind('\nreturn tables')]
markers=['selected = predictions.merge', 'development_end =', 'recursive = []',
         'us_history =', 'tables =', 'metadata =']
positions=[0]+[body.index(m) for m in markers]+[len(body)]
blocks=[body[a:b].rstrip() for a,b in zip(positions,positions[1:])]
# Put narrative comments with the next cell, not after a previous cell's display.
for i in range(len(blocks)-1):
    lines=blocks[i].splitlines(); comments=[]
    while lines and lines[-1].startswith('#'):
        comments.insert(0,lines.pop())
    blocks[i]='\n'.join(lines)
    if comments:blocks[i+1]='\n'.join(comments)+'\n'+blocks[i+1]
run_explanations=[
    ('6a. Build training rows and evaluate every candidate',
     'Each region gets a separate monthly learning table. The first 12 months supply lag features. `evaluate` runs the ten chronological folds and the final 24-month evaluation for all candidates; its full implementation is immediately above.'),
    ('6b. Choose the national one-month policy',
     'First combine regional winners, then compare that combination with every common model family. Only rows labeled `cv` enter the national choice. Refit the selected policy on all history for future forecasts; this refit does not change evaluation predictions.'),
    ('6c. Select a separate twelve-month policy',
     'Run each candidate through six complete, disjoint development-year forecasts. Inside each path, future observed stocks and flows are unavailable. Choose the lowest national development MAE and refit for the outlook.'),
    ('6d. Evaluate recursive forecasts on the final period',
     'At each of thirteen monthly origins, refit using available history and generate a full year. Compare the long-horizon selection, the one-month selection used recursively, persistence, and the original regression. The overlapping errors are correlated.'),
    ('6e. Assemble historical benchmarks and the latest forecast',
     'Sum complete PADD histories, reshape the JODI snapshot, and join by month. JODI is context only. The latest forecast combines five regional rows and one U.S. row.'),
    ('6f. Save predictions, scores, coefficients, and models',
     'Every displayed result is built from the frames above. The exports preserve individual predictions as well as aggregate scores, so another reader can check the metrics. A baseline needs its rule and historical inputs, not a fitted sklearn estimator.'),
    ('6g. Record the experiment configuration',
     'Save units, dates, feature/model settings, policy choices, software versions, and limitations alongside the results. These are generated from this run, not loaded from an earlier result.')]
run_displays=[
    "display(fold_metrics.groupby('model').agg(folds=('fold','count'),test_mae_kb=('mae_kb','mean')))\ndisplay(selection)",
    "display(policy_scores.sort_values('mae_kb'))\nprint('One-month policy:', policy)",
    "display(horizon_policy_scores)\nprint('Twelve-month policy:', horizon_policy)",
    "display(metric_table(us_recursive, ['model']))",
    "display(us_history.tail(3))\ndisplay(latest[['padd','padd_name','month','stock_kb']])",
    "print('Saved',len(tables),'result tables to',output)",
    "print('Data through:',metadata['data_end'])\nprint('Fitting warnings:',len(fit_warnings))"]
expanded=[]
for i,c in enumerate(cells):
    if i==0:
        c.source += "\n\n**Everything needed to understand the work is below:** source handling, full feature and model code, a worked validation example, the complete experiment in seven visible stages, diagnostics, conclusions, and checks. No project `.py` imports or external review document are needed. Run All uses the accompanying `data/` snapshots; all result tables and figures are already saved in this notebook."
        expanded.append(c)
        expanded.append(markdown('''### Reading guide

1. **Understand the inputs:** Sections 1–3 explain the review, series definitions, missing values, SPR, and balance checks.
2. **Understand the models:** Sections 4–5 show exactly how a forecast row and each candidate are built.
3. **Follow the experiment:** Section 6 runs the full validation and forecast workflow step by step.
4. **Read the evidence:** Sections 7–11 compare results and expose where the models fail.
5. **Reproduce and check:** Section 12 contains the conclusions and integrity checks.

Code cells define the actual functions used below; they are not pseudocode. You can read the prose and saved outputs first, then inspect each implementation in place.'''))
        continue
    if i==1:
        expanded.append(python(setup));continue
    if i==4:
        expanded.extend([
          markdown('''### Series map and input preparation

This is the exact EIA code map and sign convention. `load_panel` reshapes source observations into a complete monthly PADD panel, retains zero assumptions, derives commercial/SPR scope, and calculates historical balance residuals.'''),
          python(data_source[data_source.index('PADD_NAMES ='):data_source.index('\n\ndef refresh_data')]),
          python(definition(data_source,'load_panel')),
          markdown('''### Optional data collection, included for completeness

The downloader below preserves the XLS source bytes and their hashes, normalizes EIA observations, and snapshots JODI from DuckDB. It is defined here but **not called** during the normal offline run. Set `REFRESH_SOURCES=True` only when you want fresh EIA data and have the local JODI database available. Changing the snapshot may change dates and results.'''),
          python(definition(data_source,'refresh_data')),
          python("REFRESH_SOURCES = False\nif REFRESH_SOURCES:\n    refresh_data(HERE/'data', database=DATABASE)")])
    if i==11:
        expanded.extend([
          markdown('''### Build a forecast row from past observations

`forecast_flows` converts monthly volumes to daily rates, fits trend and calendar effects on earlier observations, then converts forecasts back to monthly volumes. `feature_row` reads only the supplied history. `supervised` stops that history immediately before each target month and attaches the target separately.'''),
          python(definitions(model_source,'seasonal_design','forecast_flows')),
          python(definitions(model_source,'feature_row','supervised'))])
    if i==13:
        expanded.extend([
          markdown('''### Exact model definitions

These constants specify the feature sets. `estimator` declares every learned model and its settings. `fit` chooses stock level or stock change as the target; `predict` converts changes back into levels and implements the simple baselines.'''),
          python(model_source[model_source.index('BASELINES ='):model_source.index('\n\ndef seasonal_design')]),
          python(definitions(model_source,'columns','estimator')),
          python(definitions(model_source,'fit','predict')),
          markdown('''### How errors and national totals are calculated

MAE averages absolute errors; RMSE squares errors before averaging and taking a square root; R² compares squared errors with variation around the evaluation sample mean. `aggregate` checks five distinct PADDs for every key before summing—regional MAEs are never added to obtain national MAE.'''),
          python(definitions(model_source,'score','aggregate','metric_table')),
          markdown('''### One worked fold, before the full comparison

This small example trains the original constrained model for PADD 3 on the first development training window, predicts the following year, and shows each error. It uses the same `fit`, `predict`, and `score` functions as the full experiment.'''),
          python("""worked = supervised(panel[panel.padd.eq(3)].reset_index(drop=True))
development_example = worked.iloc[:-24]
train_idx, test_idx = next(TimeSeriesSplit(n_splits=10, test_size=12).split(development_example))
train_example, test_example = development_example.iloc[train_idx], development_example.iloc[test_idx]
example_model, example_warning = fit('legacy_constrained_level', train_example)
example_predictions = predict('legacy_constrained_level', example_model, test_example)
worked_results = test_example[['origin_month','month','actual_kb']].copy()
worked_results['predicted_kb'] = example_predictions
worked_results['absolute_error_kb'] = abs(worked_results.actual_kb-worked_results.predicted_kb)
print('Training targets:',train_example.month.min().date(),'to',train_example.month.max().date())
display(worked_results)
display(pd.Series(score(test_example.actual_kb,example_predictions),name='Worked-fold scores'))
assert train_example.month.max() < test_example.month.min()
""")])
    if i==14:
        c.source=c.source.replace('This cell fits and evaluates all candidates from the saved inputs.','The following cells fit and evaluate all candidates from the saved inputs.')
    if i==15:
        expanded.extend([
          markdown('''### The chronological evaluation loop

The implementation below fits transformations and models separately in each training fold, collects every out-of-fold prediction, chooses regional winners from development errors, and scores the final period without using it to choose a winner. Warning messages are retained.'''),
          python(definition(model_source,'evaluate')),
          markdown('''### The recursive forecast implementation

For a year-long path, forecast all flows using history at the origin. Then predict one stock at a time and append that predicted month to the history used for the next step. The raw balance path and statistical stock path remain separate, with an explicit reconciliation difference.'''),
          python(definition(model_source,'forecast')),
          python("data_dir=HERE/'data'\noutput_dir=OUTPUT\nfolds=10")])
        for block,(title,explanation),display_code in zip(blocks,run_explanations,run_displays):
            expanded.extend([markdown('### '+title+'\n\n'+explanation),python(block+'\n\n'+display_code)])
        continue
    if i==32:
        c.source=c.source.replace('`MODEL_REVIEW.md` records findings and limitations.', 'All review findings, model implementations, and limitations are included above; the companion review document is optional.')
        c.source=c.source.replace('Run All reproduces the experiment and charts.', 'Run All executes the implementations in this notebook and reproduces the experiment and charts without importing any local Python module. Share this notebook with its `data/` folder to reproduce offline; saved outputs can be read without running code.')
        expanded.append(c)
        expanded.append(markdown('''### Reproducibility checks included in the notebook

Check source hashes, the stock decomposition, forecast timing, national aggregation, and separation of development targets from the final evaluation period. These checks run on the same inputs and outputs just used for the analysis.'''))
        expanded.append(python("""for entry in json.loads((HERE/'data/source_manifest.json').read_text()):
    source_bytes=(HERE/'data/raw'/f"{entry['series_id']}m.xls").read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest()==entry['sha256']
np.testing.assert_allclose(panel.stock_kb,panel.commercial_stock_kb+panel.spr_stock_kb)
check_history=panel[panel.padd.eq(3)].iloc[:40].copy()
before=supervised(check_history)
check_history.loc[check_history.index[-1],FLOWS+['stock_kb']]+=100000
changed=supervised(check_history)
feature_columns=[c for c in before if c not in ['actual_kb','delta_kb']]
pd.testing.assert_frame_equal(before[feature_columns],changed[feature_columns])
np.testing.assert_allclose(tables['us_forecast_12m'].stock_kb,
    tables['padd_forecast_12m'].groupby('month').stock_kb.sum())
print('Source integrity, stock scope, target-month isolation, and aggregation checks passed.')
"""))
        continue
    expanded.append(c)
cells=expanded
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nbf.write(nb,HERE/'us_snd_model_results.ipynb')
print('Created',len(cells),'cells')
