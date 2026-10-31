"""Rebuild all tables/figures from saved real runs; never starts optimization."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .runner import ROOT, read_json, write_json, write_csv
from .problems import Knapsack
from .exact import solve_exact

LABELS = {'ga_small_sigma': 'ГА: σ=0.015', 'ga_large_sigma': 'ГА: σ=0.060',
          'random': 'Случайный поиск', 'penalty_bit': 'Штраф + bit',
          'repair_bit': 'Ремонт + bit', 'penalty_swap': 'Штраф + swap',
          'random_feasible': 'Случайный допустимый'}
COLORS = ['#156b95', '#d17b21', '#77932d', '#905e9c']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .2, 'savefig.bbox': 'tight'})


def statistics(values, minimize):
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Empty or invalid results')
    return {'best': float(x.min() if minimize else x.max()), 'mean': float(x.mean()),
            'median': float(np.median(x)), 'std': float(x.std(ddof=1)) if len(x) > 1 else None,
            'worst': float(x.max() if minimize else x.min())}


def trajectory(improvements, grid):
    ticks = np.array([p['evaluations'] for p in improvements])
    values = np.array([p['best'] for p in improvements], dtype=float)
    if len(ticks) == 0:
        return np.full(len(grid), np.nan)
    idx = np.searchsorted(ticks, grid, side='right') - 1
    out = values[np.maximum(idx, 0)]
    out[idx < 0] = np.nan
    return out


def savefig(fig, directory, name):
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{name}.{ext}', dpi=170)
    plt.close(fig)


def analyze(lab):
    folder = ROOT / f'results/lab{lab}'
    manifest = read_json(folder / 'manifest.json')
    plan = manifest['plan']
    target = folder / 'analysis'
    figures = target / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    groups, histories, summaries, flat = {}, {}, [], []
    for name in plan['configs']:
        runs, hs = [], []
        for seed in plan['seeds']:
            d = folder / 'runs' / name / str(seed)
            r = read_json(d / 'result.json')
            if r['status'] != 'completed' or r['evaluations'] != plan['common']['budget']:
                raise ValueError(f'Incomplete series: {d}')
            if not r['has_feasible_solution']:
                raise ValueError(f'No feasible solution: {d}; report separately before aggregation')
            runs.append(r)
            hs.append(read_json(d / 'history.json'))
            flat.append({k: r[k] for k in ('config_id','seed','best','evaluations','elapsed_seconds',
                         'evaluated_feasible_fraction','repair_calls','generations','status')})
        groups[name], histories[name] = runs, hs
        summaries.append({'config_id': name, 'n': len(runs), **statistics([r['best'] for r in runs], lab == 1),
                          'mean_seconds': float(np.mean([r['elapsed_seconds'] for r in runs])),
                          'mean_evaluated_feasible_fraction': float(np.mean([r['evaluated_feasible_fraction'] for r in runs])),
                          'final_feasible_runs': sum(r['has_feasible_solution'] for r in runs),
                          'evaluations_per_run': plan['common']['budget']})
    write_csv(target / 'runs.csv', flat)
    write_csv(target / 'summary.csv', summaries)
    report = {'lab': lab, 'manifest': manifest, 'summary': summaries,
              'planned_runs': len(plan['seeds']) * len(plan['configs']), 'completed_runs': len(flat),
              'failed_runs': 0, 'total_evaluations': sum(r['evaluations'] for r in flat)}
    # Use exact improvement timestamps for equal-budget comparison.
    grid = np.linspace(plan['common']['population'], plan['common']['budget'], 250).astype(int)
    fig, ax = plt.subplots(figsize=(7.7, 4.2))
    for (name, runs), color in zip(groups.items(), COLORS):
        ys = np.array([trajectory(read_json(folder / 'runs' / name / str(r['seed']) / 'improvements.json'), grid) for r in runs])
        ax.plot(grid, ys.mean(0), label=LABELS[name], color=color, lw=2)
        ax.fill_between(grid, ys.min(0), ys.max(0), alpha=.10, color=color)
    ax.set(xlabel='Число вычислений целевой функции', ylabel='Лучшее найденное значение')
    if lab == 1: ax.set_yscale('log')
    ax.legend(fontsize=9)
    savefig(fig, figures, 'convergence_budget')
    fig, ax = plt.subplots(figsize=(7.7, 4.2))
    for (name, hs), color in zip(histories.items(), COLORS):
        if plan['configs'][name]['algorithm'] != 'ga': continue
        ys = np.array([[s['best'] for s in h] for h in hs], dtype=float)
        xs = [s['generation'] for s in hs[0]]
        ax.plot(xs, ys.mean(0), label=LABELS[name], color=color, lw=2)
        ax.fill_between(xs, ys.min(0), ys.max(0), alpha=.12, color=color)
    ax.set(xlabel='Поколение (0 = начальная популяция)', ylabel='Лучшее найденное значение')
    if lab == 1: ax.set_yscale('log')
    ax.legend(fontsize=9)
    savefig(fig, figures, 'convergence_generations')
    fig, ax = plt.subplots(figsize=(7.7, 4.0))
    values = [[r['best'] for r in runs] for runs in groups.values()]
    boxes = ax.boxplot(values, tick_labels=[LABELS[n].replace(' + ', '\n+ ') for n in groups], patch_artist=True)
    for patch, color in zip(boxes['boxes'], COLORS): patch.set_facecolor(color); patch.set_alpha(.4)
    for i, v in enumerate(values): ax.scatter(np.full(len(v), i+1)+np.linspace(-.07,.07,len(v)), v, s=10, color='#333333', zorder=3)
    if lab == 1: ax.set_yscale('log')
    ax.set_ylabel('Итоговое лучшее значение; 20 запусков')
    savefig(fig, figures, 'distribution')
    if lab == 2:
        p = Knapsack.from_dict(read_json(ROOT / plan['data']))
        exact = solve_exact(p)
        report['exact'] = exact
        report['greedy'] = p.describe(p.construct(sorted(range(p.n), key=lambda i: (-p.values[i]/p.weights[i], i))))
        write_json(target / 'exact_reference.json', exact)
        for s in summaries:
            s['mean_gap_percent'] = 100*(exact['value']-s['mean'])/exact['value']
            s['optimal_runs'] = sum(r['best'] == exact['value'] for r in groups[s['config_id']])
        write_csv(target / 'summary.csv', summaries)
        best = max((r for rs in groups.values() for r in rs if r['config']['algorithm']=='ga'), key=lambda r:r['best'])
        report['best_run'] = best
        write_json(target / 'best_solution.json', best)
        two = np.zeros(p.n, dtype=int); two[p.edges[0]] = 1
        over = np.ones(p.n, dtype=int)
        valid = np.zeros(p.n, dtype=int); valid[0] = 1
        report['examples'] = [{'name': 'Один предмет', **p.describe(valid)},
                              {'name': 'Лучший найденный ГА набор', **best['solution']},
                              {'name': 'Конфликтная пара', **p.describe(two)},
                              {'name': 'Все предметы', **p.describe(over)}]
        write_json(target / 'examples.json', report['examples'])
        write_csv(target / 'best_items.csv', [{'id': i+1, 'weight': int(p.weights[i]), 'value': int(p.values[i])}
                    for i in np.flatnonzero(best['best_genotype'])])
        fig, ax = plt.subplots(figsize=(7.7,4))
        for (name, hs), color in zip(histories.items(), COLORS):
            ys = np.array([[s['population_feasible_fraction'] for s in h] for h in hs])
            ax.plot([s['evaluations'] for s in hs[0]], ys.mean(0), color=color, label=LABELS[name])
        ax.set(xlabel='Число оценок', ylabel='Доля допустимых особей в популяции', ylim=(-.02,1.02))
        ax.legend(fontsize=9)
        savefig(fig, figures, 'feasibility')
        fig, ax = plt.subplots(figsize=(7.7,4.6))
        angle = np.arange(p.n) * 2*np.pi/p.n
        points = np.c_[np.cos(angle), np.sin(angle)]
        for i,j in p.edges: ax.plot(points[[i,j],0],points[[i,j],1],color='#b8c1c8',alpha=.65,lw=.7,zorder=1)
        chosen = np.array(best['best_genotype'],dtype=bool)
        ax.scatter(points[:,0],points[:,1],s=220,c=np.where(chosen,'#156b95','#eeeeee'),edgecolors='#55616b',zorder=2)
        for i,(x,y) in enumerate(points): ax.text(x,y,str(i+1),ha='center',va='center',fontsize=7,color='white' if chosen[i] else 'black')
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title('Синие вершины: выбранные предметы; линии: запреты совместного выбора',fontsize=9)
        savefig(fig, figures, 'solution_graph')
    else:
        report['best_run'] = min((r for rs in groups.values() for r in rs), key=lambda r:r['best'])
        report['known_optimum'] = {'x': [1.]*7, 'value':0.0}
    comparisons = []
    pairs = [('ga_small_sigma','ga_large_sigma'),('ga_small_sigma','random')] if lab==1 else [('penalty_bit','repair_bit'),('penalty_bit','penalty_swap'),('repair_bit','random_feasible')]
    lookup = {s['config_id']: s for s in summaries}
    for a,b in pairs:
        comparisons.append({'a':a,'b':b,'mean_b_minus_a':lookup[b]['mean']-lookup[a]['mean'],
                            'median_b_minus_a':lookup[b]['median']-lookup[a]['median'],
                            'interpretation':'descriptive, no significance claim'})
    report['comparisons'] = comparisons
    write_csv(target / 'comparisons.csv', comparisons)
    write_json(target / 'report.json', report)
    print(f'lab{lab}: {len(flat)} completed runs; {report["total_evaluations"]} evaluations')
    for s in summaries: print(s)
    return report


if __name__ == '__main__':
    for lab in (1,2): analyze(lab)
