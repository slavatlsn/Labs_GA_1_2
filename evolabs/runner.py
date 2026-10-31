import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .problems import Knapsack, generate_knapsack
from .search import run_search, validate_config

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data():
    target = ROOT / 'data/knapsack.json'
    if target.exists():
        return
    p = generate_knapsack()
    write_json(target, {**p.to_dict(), 'generator': {'seed': 51724, 'n': 40,
               'weights': 'integers [5,40]', 'values': 'integers [10,120]',
               'conflict_probability': 0.08, 'capacity_fraction': 0.30, 'numpy': np.__version__}})
    write_csv(ROOT / 'data/items.csv', [{'id': i+1, 'weight': int(w), 'value': int(v)}
               for i, (w, v) in enumerate(zip(p.weights, p.values))])
    write_csv(ROOT / 'data/conflicts.csv', [{'item_i': int(i)+1, 'item_j': int(j)+1} for i, j in p.edges])


def validate_plan(plan):
    lab = plan['lab']
    s = plan['student']
    v = 1 + (17*s['N'] + 7*s['Q'] + s['Y']) % 20
    if v != 2 or s['variant'] != v:
        raise ValueError('This implementation covers variant 2 only')
    seeds = plan['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    for params in plan['configs'].values():
        validate_config(lab, {**plan['common'], **params})
    if lab == 1 and plan['common']['dimension'] != 5 + v % 6:
        raise ValueError('Dimension does not match assignment')


def campaign(plan_path, output):
    plan = read_json(plan_path)
    validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = read_json(ROOT / plan['data']) if plan['lab'] == 2 else None
    problem = Knapsack.from_dict(data) if data else None
    sources = {name: (ROOT / 'evolabs' / name).read_text(encoding='utf-8')
               for name in ('problems.py', 'search.py', 'runner.py')}
    fingerprint = digest({'plan': plan, 'data': data, 'sources': sources})
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['fingerprint'] != fingerprint:
        raise ValueError('Code/config/data changed: use a NEW output directory')
    if not manifest_path.exists():
        write_json(manifest_path, {'schema_version': 1, 'fingerprint': fingerprint, 'plan': plan,
                   'data_sha256': digest(data), 'source_sha256': digest(sources),
                   'environment': {'python': sys.version, 'numpy': np.__version__,
                                   'platform': platform.platform(), 'processor': platform.processor(),
                                   'execution': 'sequential, one process'},
                   'started_utc': datetime.now(timezone.utc).isoformat()})
        write_json(output / 'source_snapshot.json', sources)
    failures = []
    for seed in plan['seeds']:
        for name, params in plan['configs'].items():
            folder = output / 'runs' / name / str(seed)
            final = folder / 'result.json'
            if final.exists():
                r = read_json(final)
                if r['fingerprint'] == fingerprint and r['status'] == 'completed':
                    for file in ('history.json', 'improvements.json'):
                        if not (folder / file).is_file():
                            raise ValueError(f'Missing trajectory: {folder / file}')
                    print(f'SKIP lab{plan["lab"]} {name} seed={seed}', flush=True)
                    continue
            cfg = {**plan['common'], **params}
            try:
                result, history, improvements = run_search(plan['lab'], cfg, seed, problem)
                result.update(config_id=name, fingerprint=fingerprint)
                write_json(folder / 'history.json', history)
                write_json(folder / 'improvements.json', improvements)
                write_json(final, result)
                print(f'OK lab{plan["lab"]} {name} seed={seed} best={result["best"]} '
                      f'evals={result["evaluations"]} time={result["elapsed_seconds"]:.2f}s', flush=True)
            except Exception as exc:
                import traceback
                write_json(folder / 'error.json', {'type': type(exc).__name__, 'message': str(exc),
                                                  'traceback': traceback.format_exc()})
                failures.append(f'{name}/{seed}')
                print(f'FAILED {name}/{seed}: {exc}', flush=True)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


def main():
    parser = argparse.ArgumentParser(description='ЛР1/ЛР2, группа 517, вариант 2')
    parser.add_argument('--lab', type=int, choices=[1, 2], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    prepare_data()
    campaign(args.config or ROOT / f'configs/lab{args.lab}.json',
             args.output or ROOT / f'results/lab{args.lab}')


if __name__ == '__main__': main()
