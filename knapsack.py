from ortools.algorithms.python import knapsack_solver
from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE
from random import randint
from typing import List
import pandas as pd

N_ITEMS = 30


def report_solution(packed_weights: list, packed_items: list, packed_values: list) -> None:
    print(f"Total value: ", sum(packed_values))
    print("Total weight: ", sum(packed_weights))
    print("Packed items: ", packed_items)
    print("Packed values: ", packed_values)
    print("Packed weights: ", packed_weights)


def solve_branch_and_bound(
        values: List[int], weights: List[list], capacities: List[int]
) -> None:
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    solver.init(profits=values, weights=weights, capacities=capacities)
    solver.solve()

    packed_items, packed_values, packed_weights = [], [], []

    for i in range(len(values)):
        if not solver.best_solution_contains(i):
            continue

        packed_items.append(i)
        packed_values.append(values[i])
        packed_weights.append(weights[0][i])

    report_solution(packed_weights=packed_weights, packed_values=packed_values, packed_items=packed_items)


def solve_cp_sat_single(
        values: List[int], weights: List[list], capacities: List[int]
) -> None:
    assert (
            len(weights) == 1 and len(capacities) == 1 and len(weights[0]) == len(values)
    ), "Dimensions of arguments do not match"

    model = CpModel()
    df = pd.DataFrame(data={"weight": weights[0], "value": values})
    bs = model.new_bool_var_series("b", df.index)
    model.add(bs @ df["weight"] <= capacities[0])
    model.maximize(bs @ df["value"])

    solver = CpSolver()
    status = solver.solve(model)

    assert status in (OPTIMAL, FEASIBLE)

    packed_items, packed_vals, packed_weights = [], [], []
    solved_bs = solver.values(bs)

    for i, val in enumerate(values):
        if not solved_bs[i]:
            continue
        packed_items.append(i)
        packed_vals.append(values[i])
        packed_weights.append(weights[0][i])

    report_solution(packed_weights=packed_weights, packed_values=packed_vals, packed_items=packed_items)


if __name__ == "__main__":
    values = [randint(1, 800) for _ in range(N_ITEMS)]
    weights = [[randint(1, 100) for _ in range(N_ITEMS)]]
    capacities = [850]

    solve_branch_and_bound(values=values, weights=weights, capacities=capacities)
    print("\n")
    solve_cp_sat_single(values=values, weights=weights, capacities=capacities)
