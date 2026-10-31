"""Independent reference: own branch-and-bound, not used by evolutionary search."""
from time import perf_counter
import numpy as np
from .problems import Knapsack


def solve_exact(problem: Knapsack):
    start = perf_counter()
    order = sorted(range(problem.n), key=lambda i: (-problem.values[i] / problem.weights[i], i))
    best_x = problem.construct(order)
    best = int(best_x @ problem.values)
    nodes = 0

    def visit(available, capacity, value, selected):
        nonlocal best, best_x, nodes
        nodes += 1
        if value > best:
            best = value
            best_x = np.zeros(problem.n, dtype=np.int8)
            best_x[selected] = 1
        # Fractional knapsack ignores conflicts: a valid upper bound.
        space, bound = capacity, float(value)
        for i in available:
            if problem.weights[i] <= space:
                bound += int(problem.values[i])
                space -= int(problem.weights[i])
            else:
                bound += float(problem.values[i]) * space / int(problem.weights[i])
                break
        if bound < best + 1 - 1e-9 or not available:
            return
        i, rest = available[0], available[1:]
        if problem.weights[i] <= capacity:
            visit([j for j in rest if j not in problem.adj[i]],
                  capacity - int(problem.weights[i]), value + int(problem.values[i]), selected + [i])
        visit(rest, capacity, value, selected)

    visit(order, problem.capacity, 0, [])
    return {**problem.describe(best_x), 'nodes': nodes,
            'elapsed_seconds': perf_counter() - start, 'proven_optimal': True,
            'method': 'own branch-and-bound with fractional knapsack upper bound'}
