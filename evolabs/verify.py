"""Independent audit of every saved solution and selected full replays."""
import numpy as np
from .runner import ROOT, read_json, write_json, digest
from .problems import Knapsack
from .search import run_search


def verify():
    checked, replays = 0, []
    p = Knapsack.from_dict(read_json(ROOT / 'data/knapsack.json'))
    for lab in (1,2):
        folder = ROOT / f'results/lab{lab}'
        m = read_json(folder / 'manifest.json')
        plan = m['plan']
        data = read_json(ROOT / plan['data']) if lab == 2 else None
        sources = {n: (ROOT / 'evolabs' / n).read_text() for n in ('problems.py','search.py','runner.py')}
        assert digest({'plan':plan,'data':data,'sources':sources}) == m['fingerprint'], 'Provenance mismatch'
        for name in plan['configs']:
            for seed in plan['seeds']:
                base = folder / 'runs' / name / str(seed)
                r = read_json(base / 'result.json')
                x = r['best_genotype']
                assert r['status'] == 'completed' and r['fingerprint'] == m['fingerprint']
                assert r['config'] == {**plan['common'], **plan['configs'][name]}
                assert r['seed'] == seed and r['evaluations'] == plan['common']['budget']
                assert len(x) == (7 if lab == 1 else p.n)
                if lab == 1:
                    assert all(-5 <= a <= 10 for a in x)
                    val = sum(100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2 for i in range(6))
                    assert abs(val-r['best']) < 1e-8
                else:
                    assert all(a in (0,1) for a in x)
                    assert sum(int(w)*a for w,a in zip(p.weights,x)) <= p.capacity
                    assert all(x[i]+x[j] <= 1 for i,j in p.edges)
                    assert sum(int(v)*a for v,a in zip(p.values,x)) == r['best']
                h = read_json(base / 'history.json')
                inc = read_json(base / 'improvements.json')
                assert h[0]['evaluations'] == plan['common']['population']
                assert h[-1]['evaluations'] == r['evaluations'] and h[-1]['best'] == r['best']
                assert inc[-1]['best'] == r['best']
                assert all(a['evaluations'] < b['evaluations'] for a,b in zip(inc,inc[1:]))
                assert inc[-1]['evaluations'] <= r['evaluations']
                checked += 1
        # Reproduce the first full-budget run of each algorithm, ignoring wall time.
        for name in plan['configs']:
            seed = plan['seeds'][0]
            original = read_json(folder / 'runs' / name / str(seed) / 'result.json')
            fresh, _, inc = run_search(lab, original['config'], seed, p if lab==2 else None)
            assert fresh['best_genotype'] == original['best_genotype']
            assert inc == read_json(folder / 'runs' / name / str(seed) / 'improvements.json')
            replays.append({'lab':lab,'config':name,'seed':seed,'identical_solution_and_improvements':True})
    exact = read_json(ROOT / 'results/lab2/analysis/exact_reference.json')
    assert exact['feasible'] and exact['proven_optimal']
    for f in (ROOT / 'results/lab2/runs').glob('*/*/result.json'):
        assert read_json(f)['best'] <= exact['value']
    result = {'status':'passed','checked_saved_runs':checked,'full_budget_replays':replays,
              'all_budgets_exact':True,'independent_solution_checks':True,
              'note':'Validation and exact-reference work are outside the search budgets.'}
    write_json(ROOT / 'results/verification.json', result)
    print(result)


if __name__ == '__main__': verify()
