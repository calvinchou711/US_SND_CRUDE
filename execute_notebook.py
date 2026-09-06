"""Execute the generated walkthrough, show progress, and retain partial output on failure."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-saved', action='store_true',
                        help='Refresh reporting from existing CSVs while retaining saved experiment outputs.')
    args = parser.parse_args()
    path = Path(__file__).resolve().with_name('us_snd_model_results.ipynb')
    notebook = nbformat.read(path, as_version=4)
    client = ProgressClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(path.parent)}})
    try:
        if not args.render_saved:
            client.execute()
        else:
            start = next(i for i,c in enumerate(notebook.cells) if c.source.startswith('output = Path(output_dir)'))
            end = next(i for i,c in enumerate(notebook.cells) if c.source.startswith("folds=tables['fold_metrics']"))
            with client.setup_kernel():
                for i, cell in enumerate(notebook.cells):
                    if cell.cell_type == 'code' and start <= i < end:
                        cell.outputs = []
                        cell.execution_count = None
                    if i == end:
                        setup = nbformat.v4.new_code_cell('''
metadata = json.loads((OUTPUT/'model_metadata.json').read_text())
assert metadata['hyperparameter_search']['folds'] == 10
tables = {p.stem: pd.read_csv(p) for p in OUTPUT.glob('*.csv')}
for table in tables.values():
    if 'model' in table:
        table.drop(table.index[table.model.eq('legacy_constrained_level')], inplace=True)
    for column in ['month', 'origin_month', 'train_end', 'test_start', 'test_end']:
        if column in table:
            table[column] = pd.to_datetime(table[column])
''')
                        notebook.cells.append(setup)
                        try:
                            client.execute_cell(setup, len(notebook.cells) - 1)
                        finally:
                            notebook.cells.pop()
                    if cell.cell_type == 'code' and not start <= i < end:
                        client.execute_cell(cell, i)
    finally:
        nbformat.write(notebook, path)
    print('Notebook reporting refreshed from saved results.' if args.render_saved else
          f'Executed {len(notebook.cells)} cells successfully.')
