# Crude-model review

The findings and numerical results below describe the **pre-grid-search review**.
The September 6 update adds ten-fold chronological GridSearchCV to the current
model and notebook. See [README.md](README.md#parameter-grids) for the ranges and
protocol, and the current notebook and `model_output/` for tuned results. The
fixed-setting outputs referenced below are preserved in
`model_output_before_grid_search_20260906/`.

Reviewed the original Python model, all three notebooks, generated outputs, source
series, aggregation logic, and tests. Preserved the pre-review files under
`legacy_results_20260905/` before changing the current implementation.

## Correctness findings implemented

1. **Accounting timing.** The old formula adds month-t flows to end-t stocks and
   calls the result an identity for t+1. Actual month-t flows explain the change
   from end-(t−1) to end-t. Current code exposes this historical identity separately
   from forecasts. The original regression's t-to-t+1 lagged-flow specification is
   preserved as a comparison; using those lagged observations is not itself leakage.
2. **Incomplete regional balance.** Downloaded net receipts, adjustments, transfers,
   and direct use. The old residual reached tens of millions of barrels regionally.
   Corrected accounting generally closes within rounding, with visible January 2026
   60/6-kb stock-level versus reported-change exceptions.
3. **SPR scope.** Original stock keys include the reserve. Downloaded non-SPR stocks;
   retained the original total target and showed the commercial/reserve decomposition.
4. **Forecast consistency.** One-step features, recursive paths, clipping, flow units,
   and aggregation use the same functions. Flow regressions operate on daily rates
   before conversion to monthly volumes. Signed receipts and adjustments remain signed.
5. **Data integrity.** Frozen XLS/CSV inputs and hashes, missing-flow flags, conflicting
   duplicate rejection, complete monthly calendars, and rejection of partial U.S. sums.
   Fixed Excel parsing in both crude and gasoline downloaders so literal `NA` is not
   mistaken for a blank sparse-flow cell. Current snapshots contain no such markers
   in the analysis period, so gasoline results are unchanged.
6. **Honest reconciliation.** Forecast flow balances and statistical stock changes
   need not agree; their difference is exported separately from observed adjustments.
7. **Legacy output protection.** Preserved original results and marked older comparison
   notebooks. Updated coefficient outputs so current results cannot silently retain
   old coefficients. Archived tests are excluded from ordinary pytest discovery.

## Improvements tested

Compared persistence, last-year stock, trailing seasonal stock change, forecast-flow
accounting, original and complete constrained regressions, ridge/seasonal ridge,
polynomial/spline ridge, random forest, XGBoost, neural network, a 60-month ridge,
a ridge shrunk halfway to persistence, and an equal-weight ridge/forest/XGBoost ensemble.

All candidate settings are fixed, transformations fit within training data, and
fitting warnings are saved. Candidate specifications are in `candidate_models.json`.

Ten expanding one-month validation folds select regional winners and a national
aggregation policy. All candidates are evaluated on a subsequent 24-month period.
The original constrained regression is refitted on identical dates for comparison.
The neural network no longer suppresses convergence warnings or relies on randomized
within-training early-stopping validation.

The seasonal-change national rule wins development selection. Its one-month final
MAE is 11,662 kb versus 13,182 kb for persistence and 11,670 kb for the old regression.
More complicated models can look better in the final table, but selecting one using
that table would turn evaluation data into tuning data. They are not substituted.

## Long-horizon failure and implications

After one-step selection exposed poor year-long behavior, this review added a
separate recursive selection protocol: six disjoint development forecast years,
ending before the final evaluation period, with all 16 candidates compared.
Seasonal change still wins that development comparison (37,515-kb MAE versus
47,529 kb for persistence), so the selected policy is unchanged.

The 13 overlapping final-period paths show MAE of 39,854 kb for seasonal change,
24,053 kb for the original regression, and 14,931 kb for persistence. This is a
failed generalization result, not evidence of improved 12-month skill. The paths
are correlated and no significance or calibrated interval claim is made.

Prior SPR drawdowns make a historical seasonal rule extrapolate a policy regime
that need not persist. A future research project should model commercial stocks
separately and evaluate explicit, dated SPR scenarios. This review does not assume
an unobserved government policy or tune a new rule to the final-period failures.
The selected forecast is retained for auditability alongside a visible persistence
baseline, with clear limits on its interpretation.

## Remaining limits

- Latest-vintage monthly observations, not historical release vintages. The forecast
  origin means the latest observed monthly period, not the date the data was released.
- The final period is held out of this experiment's selection; earlier notebooks
  used some of those dates. This is retrospective model review, not a prospective trial.
- The long-horizon protocol was added during the review after examining initial
  recursive behavior. Its ranking uses development data only; it is not described
  as a completely blind end-to-end experiment.
- Reporting-category changes and sparse-flow zero assumptions remain explicit.
- JODI provides national context with differing coverage and revision dates, not
  independent regional observations or features used to predict stocks.
- No calibrated uncertainty bands or validated price/trading claims.

## Sources

- [EIA balance definitions](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp)
- [EIA PADD 3 balance](https://www.eia.gov/dnav/pet/pet_sum_snd_d_r30_mbbl_m_cur.htm)
- [EIA non-SPR stock series](https://www.eia.gov/dnav/pet/pet_stoc_typ_a_epc0_sax_mbbl_m.htm)
- [JODI source guide](https://www.jodidata.org/oil/support/user-guide/data-available-in-the-jodi-oil-world-database.aspx)

## Verification

The executed notebook has 72 cells and nine saved plots, with no cell errors.
All 13 crude tests (including the preserved original model) pass, and all six
gasoline tests pass after the parser correction. Source hashes, monthly alignment,
SPR decomposition, same-date legacy regression equivalence, target/future-feature
isolation, five-region aggregation, development-only policy selection, recursive
path dates, and forecast accounting were checked. The four CSVs consumed by the
database builder load successfully through DuckDB.

A workspace-root pytest invocation cannot collect the entire multi-project
workspace: separate packages are not installed/importable from that root and
several test modules share names. Tests above were run separately from each model
directory, as documented. No claim is made that the whole workspace suite passes.

The consolidated notebook now includes the full data collector/loader, feature
engineering, candidate definitions, scoring and aggregation, chronological
evaluation loop, recursive forecast routine, and experiment/export code in ordinary
visible cells. A worked fold precedes the full experiment. No local Python module
or companion review document is required to follow or execute the analysis.
