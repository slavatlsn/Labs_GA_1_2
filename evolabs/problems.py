from dataclasses import dataclass
import numpy as np


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite matrix (batch, dimension >= 2)')
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
                  + (x[:, :-1] - 1) ** 2, axis=1)


def reflect(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    width = upper - lower
    if width <= 0:
        raise ValueError('upper must exceed lower')
    phase = (np.asarray(x) - lower) % (2 * width)
    return lower + np.where(phase <= width, phase, 2 * width - phase)


@dataclass
class Knapsack:
    weights: np.ndarray
    values: np.ndarray
    capacity: int
    edges: np.ndarray

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.int64)
        self.values = np.asarray(self.values, dtype=np.int64)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        n = len(self.weights)
        if n == 0 or self.values.shape != (n,) or self.weights.shape != (n,):
            raise ValueError('Invalid item arrays')
        if (self.weights <= 0).any() or (self.values <= 0).any() or self.capacity <= 0:
            raise ValueError('Weights, values and capacity must be positive')
        if self.edges.size and ((self.edges < 0).any() or (self.edges >= n).any()
                                or (self.edges[:, 0] == self.edges[:, 1]).any()):
            raise ValueError('Invalid conflict edge')
        canonical = {tuple(sorted(e)) for e in self.edges.tolist()}
        if len(canonical) != len(self.edges):
            raise ValueError('Duplicate conflicts')
        self.adj = [set() for _ in range(n)]
        for i, j in self.edges:
            self.adj[i].add(int(j))
            self.adj[j].add(int(i))

    @property
    def n(self):
        return len(self.weights)

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.n or not np.isin(x, [0, 1]).all():
            raise ValueError('Expected binary matrix')
        value = x @ self.values
        weight = x @ self.weights
        conflicts = np.sum(x[:, self.edges[:, 0]] * x[:, self.edges[:, 1]], axis=1)
        excess = np.maximum(0, weight - self.capacity)
        feasible = (excess == 0) & (conflicts == 0)
        return value, feasible, excess, conflicts

    def repair(self, x):
        """Delete lower value/weight endpoint, then lightest-priority excess items.

        Uses only item coefficients and constraints, never complete objective calls.
        Ties remove the greater item index. No greedy addition/local optimization.
        """
        x = np.asarray(x, dtype=np.int8).copy()
        ratio = self.values / self.weights
        for i, j in self.edges:
            if x[i] and x[j]:
                remove = i if (ratio[i], -int(i)) < (ratio[j], -int(j)) else j
                x[remove] = 0
        weight = int(x @ self.weights)
        for i in sorted(np.flatnonzero(x), key=lambda k: (ratio[k], -int(k))):
            if weight <= self.capacity:
                break
            x[i] = 0
            weight -= int(self.weights[i])
        return x

    def construct(self, order):
        """Feasible constructive baseline, respecting capacity and conflicts."""
        x = np.zeros(self.n, dtype=np.int8)
        selected = set()
        weight = 0
        for i in order:
            i = int(i)
            if i not in selected and weight + self.weights[i] <= self.capacity and not (self.adj[i] & selected):
                selected.add(i)
                x[i] = 1
                weight += int(self.weights[i])
        return x

    def describe(self, x):
        value, feasible, excess, conflicts = self.evaluate(np.asarray(x)[None, :])
        return {'genotype': np.asarray(x, dtype=int).tolist(),
                'selected_ids': (np.flatnonzero(x) + 1).tolist(),
                'value': int(value[0]), 'weight': int(np.asarray(x) @ self.weights),
                'capacity': self.capacity, 'feasible': bool(feasible[0]),
                'excess_weight': int(excess[0]), 'conflict_count': int(conflicts[0])}

    def to_dict(self):
        return {'weights': self.weights.tolist(), 'values': self.values.tolist(),
                'capacity': self.capacity, 'edges_zero_based': self.edges.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d['weights'], d['values'], d['capacity'], d['edges_zero_based'])


def generate_knapsack(seed=51724, n=40, conflict_probability=0.08):
    rng = np.random.default_rng(seed)
    weights = rng.integers(5, 41, n)
    values = rng.integers(10, 121, n)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if rng.random() < conflict_probability]
    return Knapsack(weights, values, int(0.30 * sum(weights)), np.array(edges))
