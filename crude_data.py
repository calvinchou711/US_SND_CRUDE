"""EIA monthly supply, disposition, and inventory data."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
PADD_NAMES = {1: 'East Coast', 2: 'Midwest', 3: 'Gulf Coast', 4: 'Rocky Mountain', 5: 'West Coast'}
CODES = {'production_kbd': 'MCRFPP{p}1', 'demand_kbd': 'MCRRIP{p}1',
         'imports_kbd': 'MCRIMP{p}1', 'exports_kbd': 'MCREXP{p}1',
         'net_receipts_kbd': 'MCRNRP{p}1', 'adjustments_kbd': 'MCRUA_R{p}0_1',
         'transfers_kbd': 'M_EPC0_TVP_R{p}0_MBBL', 'direct_use_kbd': 'MCRUPP{p}1',
         'stock_kb': 'MCESTP{p}1', 'cdu_capacity_kbd': 'MOCLEP{p}2'}
FLOWS = ['production_kbd', 'demand_kbd', 'imports_kbd', 'exports_kbd',
         'net_receipts_kbd', 'adjustments_kbd', 'transfers_kbd', 'direct_use_kbd']
SIGNS = [1, -1, 1, -1, 1, 1, 1, -1]
# The downloaded transfer series start in January 2022; earlier rows are structural zeros.
SPARSE = {'exports_kbd', 'transfers_kbd', 'net_receipts_kbd'}


def refresh_data(data_dir=HERE / 'data'):
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
        title = str(table.columns[1])
        if component == 'cdu_capacity_kbd' and 'Thousand Barrels per Calendar Day' not in title:
            raise ValueError(f'Unexpected CDU capacity units: {title}')
        if component in FLOWS and 'per' in title.split('(')[-1].lower():
            raise ValueError(f'Expected monthly flow volumes: {title}')
        (raw_dir / f'{code}m.xls').write_bytes(content)
        original = table.iloc[:, 1]
        values = pd.to_numeric(original, errors='coerce')
        # Excel empty cells correspond to no-data-reported on these sparse series.
        # Preserve text markers so W/NA cannot silently turn into zero.
        empty = original.isna() | original.astype(str).str.strip().isin(['', '-'])
        zero = values.isna() & empty & (component in SPARSE)
        values = values.mask(zero, 0.0)
        source_component = ('commercial_stock_kb' if component == 'stock_kb' else
                            component.replace('_kbd', '_kb') if component in FLOWS else component)
        frame = pd.DataFrame({'month': pd.to_datetime(table.iloc[:, 0]).dt.to_period('M').dt.to_timestamp(),
            'padd': p, 'component': source_component, 'value': values, 'assumed_zero': zero, 'series_id': code})
        # Sparse worksheets sometimes stop before the common archive endpoint;
        # calendar extension is handled explicitly by load_panel, with flags.
        meta = {'padd': p, 'component': source_component, 'series_id': code, 'url': url,
                'title': str(table.columns[1]), 'sha256': hashlib.sha256(content).hexdigest(),
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
        return frame, meta
    tasks = [(p, c, template.format(p=p)) for p in PADD_NAMES for c, template in CODES.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(download, tasks))
    raw = pd.concat([r[0] for r in results], ignore_index=True)
    raw.to_csv(data_dir / 'eia_observations.csv', index=False)
    (data_dir / 'source_manifest.json').write_text(json.dumps([r[1] for r in results], indent=2))
    return load_panel(data_dir)


def load_panel(data_dir=HERE / 'data', start='2007-01-01'):
    raw = pd.read_csv(Path(data_dir) / 'eia_observations.csv', parse_dates=['month'])
    # Raw flow snapshots retain EIA monthly volumes; convert exactly once here.
    raw['component'] = raw.component.replace({c.replace('_kbd', '_kb'): c for c in FLOWS})
    raw = raw[raw.component.isin(FLOWS + ['commercial_stock_kb', 'cdu_capacity_kbd'])].copy()
    raw['component'] = raw.component.replace({'commercial_stock_kb': 'stock_kb'})
    raw['value'] = raw.value.astype(float)
    is_flow = raw.component.isin(FLOWS)
    raw.loc[is_flow, 'value'] /= raw.loc[is_flow, 'month'].dt.days_in_month
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
    core = FLOWS + ['stock_kb', 'cdu_capacity_kbd']
    valid = panel[core].notna().all(axis=1)
    common = panel[valid].groupby('month').padd.nunique()
    end = common[common.eq(5)].index.max()
    panel = panel[panel.month.le(end)].copy()
    if panel[core].isna().any().any():
        raise ValueError('Unresolved missing core data; inspect the source snapshots')
    if not panel.groupby('padd').size().eq(len(pd.date_range(start, end, freq='MS'))).all():
        raise ValueError('Incomplete PADD calendar')
    if panel.cdu_capacity_kbd.le(0).any():
        raise ValueError('CDU capacity must be positive')
    panel['utilization_ratio'] = panel.demand_kbd / panel.cdu_capacity_kbd
    panel['previous_stock_kb'] = panel.groupby('padd').stock_kb.shift(1)
    panel['balance_kbd'] = panel[FLOWS].to_numpy() @ SIGNS
    panel['stock_change_kb'] = panel.stock_kb - panel.previous_stock_kb
    panel['five_term_balance_kbd'] = panel.production_kbd + panel.imports_kbd - panel.demand_kbd - panel.exports_kbd
    panel['five_term_residual_kbd'] = panel.stock_change_kb / panel.month.dt.days_in_month - panel.five_term_balance_kbd
    panel['accounting_residual_kbd'] = panel.stock_change_kb / panel.month.dt.days_in_month - panel.balance_kbd
    return panel


if __name__ == '__main__':
    panel = refresh_data()
    print(panel.groupby('padd').agg(start=('month', 'min'), end=('month', 'max'), n=('month', 'size')))
    print(panel.groupby('padd').accounting_residual_kbd.agg(['min', 'max']))
