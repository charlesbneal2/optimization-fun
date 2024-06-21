from ortools.algorithms.python import knapsack_solver
from random import randint

N_ITEMS = 30


def solve_branch_and_bound(values: list, weights: list, capacities: list) -> int:
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    solver.init(profits=values, weights=weights, capacities=capacities)
    total = solver.solve()
    print(f"Total value: {total}")

    packed_items, packed_weights = [], []

    for i in range(len(values)):
        if not solver.best_solution_contains(i):
            continue

        packed_items.append(i)
        packed_weights.append(weights[0][i])

    print("Total weight: ", sum(packed_weights))
    print("Packed items: ", packed_items)
    print("Packed weights: ", packed_weights)

    return total

if __name__ == "__main__":
    values = [randint(1, 800) for _ in range(N_ITEMS)]
    weights = [[randint(1, 100) for _ in range(N_ITEMS)]]
    capacities = [850]

    solve_branch_and_bound(values=values, weights=weights, capacities=capacities)

