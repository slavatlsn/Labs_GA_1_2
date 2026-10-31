from time import perf_counter
import numpy as np
from .problems import rosenbrock, reflect, Knapsack


class Evaluator:
    """All search objective evaluations pass here; initial population is charged."""
    def __init__(self, lab, budget, problem=None):
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError('Budget must be a positive integer')
        self.lab, self.limit, self.problem = lab, budget, problem
        self.used = 0
        self.best = None
        self.best_x = None
        self.feasible_evaluations = 0
        self.improvements = []
        self.t0 = perf_counter()

    def evaluate(self, x, generation):
        if self.used + len(x) > self.limit:
            raise ValueError('Evaluation budget would be exceeded')
        if self.lab == 1:
            values = rosenbrock(x)
            feasible = np.ones(len(x), dtype=bool)
            score = values
        else:
            values, feasible, excess, conflicts = self.problem.evaluate(x)
            # Integer violations >= 1: every feasible candidate outranks infeasible.
            score = -values + (int(self.problem.values.sum()) + 1) * (excess + conflicts)
        for i in np.flatnonzero(feasible):
            v = float(values[i]) if self.lab == 1 else int(values[i])
            better = self.best is None or (v < self.best if self.lab == 1 else v > self.best)
            if better:
                self.best, self.best_x = v, x[i].copy()
                self.improvements.append({'evaluations': self.used + int(i) + 1, 'best': v})
        self.used += len(x)
        self.feasible_evaluations += int(feasible.sum())
        return score.astype(float), values, feasible

    def snapshot(self, generation, values, feasible, population):
        good = values[feasible]
        return {'generation': generation, 'evaluations': self.used,
                'elapsed_seconds': perf_counter() - self.t0,
                'best': self.best, 'population_feasible_fraction': float(feasible.mean()),
                'population_mean_feasible': float(good.mean()) if len(good) else None,
                'unique_fraction': len(np.unique(population, axis=0)) / len(population)}


def tournament(pop, score, rng, count, k):
    indices = rng.integers(0, len(pop), (count, k))
    winners = indices[np.arange(count), np.argmin(score[indices], axis=1)]
    return pop[winners].copy()


def binary_mutate(x, rng, method):
    x = x.copy()
    if method == 'bit':
        return x ^ (rng.random(x.shape) < 1 / x.shape[1]).astype(np.int8)
    if method != 'swap':
        raise ValueError('Unknown binary mutation')
    # A true exchange: preserves count of ones; crossover may change cardinality.
    for row in x:
        i, j = rng.choice(len(row), 2, replace=False)
        row[i], row[j] = row[j], row[i]
    return x


def validate_config(lab, cfg):
    if lab not in (1, 2):
        raise ValueError('lab must be 1 or 2')
    if cfg['algorithm'] not in ('ga', 'random'):
        raise ValueError('Unknown algorithm')
    if cfg['population'] < (30 if lab == 1 else 2):
        raise ValueError('Population too small')
    if not isinstance(cfg['budget'], int) or cfg['budget'] < cfg['population']:
        raise ValueError('Budget smaller than initial population')
    if lab == 1 and (cfg['dimension'] < 2 or cfg['lower'] >= cfg['upper']):
        raise ValueError('Invalid continuous domain')
    if cfg['algorithm'] == 'ga':
        if not 1 <= cfg['elite'] < cfg['population'] or cfg['tournament'] < 1:
            raise ValueError('Invalid elitism or tournament')
        if not 0 <= cfg['crossover_probability'] <= 1:
            raise ValueError('Invalid crossover probability')
        if lab == 1 and (cfg['sigma'] <= 0 or not 0 <= cfg['mutation_probability'] <= 1):
            raise ValueError('Invalid Gaussian mutation')
        if lab == 2 and (cfg['constraints'] not in ('penalty', 'repair') or cfg['mutation'] not in ('bit', 'swap')):
            raise ValueError('Invalid discrete operators')


def run_search(lab: int, cfg: dict, seed: int, problem: Knapsack | None = None):
    validate_config(lab, cfg)
    rng = np.random.default_rng(seed)
    ev = Evaluator(lab, cfg['budget'], problem)
    size = cfg['population']
    repairs = 0

    def sample(count):
        if lab == 1:
            return rng.uniform(cfg['lower'], cfg['upper'], (count, cfg['dimension']))
        if cfg['algorithm'] == 'random':
            return np.array([problem.construct(rng.permutation(problem.n)) for _ in range(count)])
        return (rng.random((count, problem.n)) < cfg['initial_one_probability']).astype(np.int8)

    def repair_batch(x):
        nonlocal repairs
        if lab == 2 and cfg['algorithm'] == 'ga' and cfg['constraints'] == 'repair':
            repairs += len(x)
            return np.array([problem.repair(row) for row in x])
        return x

    pop = repair_batch(sample(size))
    score, values, feasible = ev.evaluate(pop, 0)
    history = [ev.snapshot(0, values, feasible, pop)]
    generation = 0
    while ev.used < ev.limit:
        generation += 1
        if cfg['algorithm'] == 'random':
            count = min(size, ev.limit - ev.used)
            pop = sample(count)
            score, values, feasible = ev.evaluate(pop, generation)
        else:
            elite = np.argsort(score, kind='stable')[:cfg['elite']]
            count = min(size - cfg['elite'], ev.limit - ev.used)
            a = tournament(pop, score, rng, count, cfg['tournament'])
            b = tournament(pop, score, rng, count, cfg['tournament'])
            crossed = rng.random(count) < cfg['crossover_probability']
            if lab == 1:
                # BLX-alpha, alpha=.25: independent coordinates in expanded interval.
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                width = hi - lo
                offspring = rng.uniform(lo - .25 * width, hi + .25 * width)
                offspring[~crossed] = a[~crossed]
                mask = rng.random(offspring.shape) < cfg['mutation_probability']
                offspring += mask * rng.normal(0, cfg['sigma'] * (cfg['upper'] - cfg['lower']), offspring.shape)
                offspring = reflect(offspring, cfg['lower'], cfg['upper'])
            else:
                offspring = np.where(rng.random(a.shape) < .5, a, b)
                offspring[~crossed] = a[~crossed]
                offspring = binary_mutate(offspring, rng, cfg['mutation'])
            offspring = repair_batch(offspring)
            new_score, new_values, new_feasible = ev.evaluate(offspring, generation)
            pop = np.concatenate([pop[elite], offspring])
            score = np.concatenate([score[elite], new_score])
            values = np.concatenate([values[elite], new_values])
            feasible = np.concatenate([feasible[elite], new_feasible])
        history.append(ev.snapshot(generation, values, feasible, pop))
    result = {'schema_version': 1, 'status': 'completed', 'stop_reason': 'budget_exhausted',
              'lab': lab, 'seed': seed, 'config': cfg, 'evaluations': ev.used,
              'elapsed_seconds': perf_counter() - ev.t0, 'best': ev.best,
              'has_feasible_solution': ev.best is not None,
              'feasible_evaluations': ev.feasible_evaluations,
              'evaluated_feasible_fraction': ev.feasible_evaluations / ev.used,
              'repair_calls': repairs, 'generations': generation,
              'best_genotype': ev.best_x.tolist() if ev.best_x is not None else None}
    if lab == 2 and ev.best_x is not None:
        result['solution'] = problem.describe(ev.best_x)
    return result, history, ev.improvements
