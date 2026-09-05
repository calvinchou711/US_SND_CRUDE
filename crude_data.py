"""Reproducible EIA downloads and a local JODI snapshot; no database writes."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import duckdb
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATABASE = HERE.parents[1] / 'data/commodities.duckdb'
PADD_NAMES = {1: 'East Coast', 2: 'Midwest', 3: 'Gulf Coast', 4: 'Rocky Mountain', 5: 'West Coast'}
CODES = {'production_kb': 'MCRFPP{p}1', 'demand_kb': 'MCRRIP{p}1',
         'imports_kb': 'MCRIMP{p}1', 'exports_kb': 'MCREXP{p}1',
         'net_receipts_kb': 'MCRNRP{p}1', 'adjustments_kb': 'MCRUA_R{p}0_1',
         'transfers_kb': 'M_EPC0_TVP_R{p}0_MBBL', 'direct_use_kb': 'MCRUPP{p}1',
         'stock_kb': 'MCRSTP{p}1', 'reported_stock_change_kb': 'MCRSCP{p}1',
         'commercial_stock_kb': 'MCESTP{p}1'}
FLOWS = ['production_kb', 'demand_kb', 'imports_kb', 'exports_kb',
         'net_receipts_kb', 'adjustments_kb', 'transfers_kb', 'direct_use_kb']
SIGNS = [1, -1, 1, -1, 1, 1, 1, -1]
# The downloaded transfer series start in January 2022; earlier rows are structural zeros.
SPARSE = {'exports_kb', 'transfers_kb', 'net_receipts_kb'}


def refresh_data(data_dir=HERE / 'data', database=DATABASE):
    data_dir = Path(data_dir)
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    def download(task):
        p, component, code = task
        url = f'https://www.eia.gov/dnav/pet/hist_xls/{code}m.xls'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = response.content
        table = pd.read_excel(io.BytesIO(content), sheet_name='Data 1', header=2, keep_default_na=False)
        if table.shape[1] != 2 or 'Thousand Barrels' not in str(table.columns[1]):
            raise ValueError(f'Unexpected units/schema: {code}')
        (raw_dir / f'{code}m.xls').write_bytes(content)
        original = table.iloc[:, 1]
        values = pd.to_numeric(original, errors='coerce')
        # Excel empty cells correspond to no-data-reported on these sparse series.
        # Preserve text markers so W/NA cannot silently turn into zero.
        empty = original.isna() | original.astype(str).str.strip().isin(['', '-'])
        zero = values.isna() & empty & (component in SPARSE)
        values = values.mask(zero, 0.0)
        frame = pd.DataFrame({'month': pd.to_datetime(table.iloc[:, 0]).dt.to_period('M').dt.to_timestamp(),
            'padd': p, 'component': component, 'value': values, 'assumed_zero': zero, 'series_id': code})
        # Sparse worksheets sometimes stop before the common archive endpoint;
        # calendar extension is handled explicitly by load_panel, with flags.
        meta = {'padd': p, 'component': component, 'series_id': code, 'url': url,
                'title': str(table.columns[1]), 'sha256': hashlib.sha256(content).hexdigest(),
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
        return frame, meta
    tasks = [(p, c, template.format(p=p)) for p in PADD_NAMES for c, template in CODES.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(download, tasks))
    raw = pd.concat([r[0] for r in results], ignore_index=True)
    raw.to_csv(data_dir / 'eia_observations.csv', index=False)
    (data_dir / 'source_manifest.json').write_text(json.dumps([r[1] for r in results], indent=2))
    with duckdb.connect(str(database), read_only=True) as con:
        jodi = con.execute("""SELECT time_period, flow_breakdown, obs_value, assessment_code
            FROM fundamental.jodi_observation WHERE ref_area='US'
            AND energy_product='CRUDEOIL' AND unit_measure='KBBL'""").fetchdf()
    jodi.to_csv(data_dir / 'jodi_us_raw.csv', index=False)
    (data_dir / 'jodi_source.json').write_text(json.dumps({'database': str(database),
        'table': 'fundamental.jodi_observation', 'product': 'CRUDEOIL', 'unit': 'KBBL',
        'snapshot_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    return load_panel(data_dir)


def load_panel(data_dir=HERE / 'data', start='2007-01-01'):
    raw = pd.read_csv(Path(data_dir) / 'eia_observations.csv', parse_dates=['month'])
    if raw.duplicated(['month', 'padd', 'component']).any():
        raise ValueError('Duplicate EIA observations')
    panels = []
    for p in PADD_NAMES:
        r = raw[raw.padd.eq(p)]
        end = r[r.component.eq('stock_kb') & r.value.notna()].month.max()
        index = pd.date_range(start, end, freq='MS')
        wide = r.pivot(index='month', columns='component', values='value').reindex(index)
        for c in SPARSE:
            rows = r[r.component.eq(c)].set_index('month')
            # Missing calendar rows in sparse export/transfer sheets are an explicit
            # no-reported-flow assumption, not interpolation from future values.
            absent = ~wide.index.isin(rows.index)
            wide[c + '_assumed_zero'] = rows.assumed_zero.reindex(index).astype("boolean").fillna(False).astype(bool) | absent
            wide.loc[absent, c] = 0.0
        wide['padd'] = p
        wide['padd_name'] = PADD_NAMES[p]
        wide.index.name = 'month'
        panels.append(wide.reset_index())
    panel = pd.concat(panels, ignore_index=True)
    core = FLOWS + ['stock_kb', 'reported_stock_change_kb', 'commercial_stock_kb']
    valid = panel[core].notna().all(axis=1)
    common = panel[valid].groupby('month').padd.nunique()
    end = common[common.eq(5)].index.max()
    panel = panel[panel.month.le(end)].copy()
    if panel[core].isna().any().any():
        raise ValueError('Unresolved missing core data; inspect the source snapshots')
    if not panel.groupby('padd').size().eq(len(pd.date_range(start, end, freq='MS'))).all():
        raise ValueError('Incomplete PADD calendar')
    panel['spr_stock_kb'] = panel.stock_kb - panel.commercial_stock_kb
    panel['previous_stock_kb'] = panel.groupby('padd').stock_kb.shift(1)
    panel['balance_kb'] = panel[FLOWS].to_numpy() @ SIGNS
    panel['stock_change_kb'] = panel.stock_kb - panel.previous_stock_kb
    panel['five_term_balance_kb'] = panel.production_kb + panel.imports_kb - panel.demand_kb - panel.exports_kb
    panel['five_term_residual_kb'] = panel.stock_change_kb - panel.five_term_balance_kb
    panel['accounting_residual_kb'] = panel.stock_change_kb - panel.balance_kb
    panel['reported_change_residual_kb'] = panel.stock_change_kb - panel.reported_stock_change_kb
    return panel


if __name__ == '__main__':
    panel = refresh_data()
    print(panel.groupby('padd').agg(start=('month', 'min'), end=('month', 'max'), n=('month', 'size')))
    print(panel.groupby('padd').accounting_residual_kb.agg(['min', 'max']))
