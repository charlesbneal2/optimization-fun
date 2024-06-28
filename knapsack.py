from ortools.algorithms.python import knapsack_solver
from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE
from pydantic import BaseModel, PositiveInt, NonNegativeFloat, model_validator
from random import randint
from typing import List
import pandas as pd

N_ITEMS = 30


class KnapsackInstance(BaseModel):
    weights: List[PositiveInt]
    values: List[PositiveInt]
    capacity: PositiveInt

    @model_validator(mode="after")
    def validate_lengths(cls, v):
        if len(v.weights) != len(v.values):
            raise ValueError("Mismatch in number of weights and values")
        return v


class KnapsackSolverConfig(BaseModel):
    time_limit: PositiveInt = 900
    opt_tol: NonNegativeFloat = 0.01


class KnapsackSolution(BaseModel):
    selected_items: List[int]
    objective: int
    upper_bound: float


def report_solution(
    packed_weights: list, packed_items: list, packed_values: list
) -> None:
    print(f"Total value: ", sum(packed_values))
    print("Total weight: ", sum(packed_weights))
    print("Packed items: ", packed_items)
    print("Packed values: ", packed_values)
    print("Packed weights: ", packed_weights)


def solve_branch_and_bound(instance: KnapsackInstance) -> None:
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    solver.init(profits=instance.values, weights=instance.weights, capacities=instance.capacities)
    solver.solve()

    packed_items, packed_values, packed_weights = [], [], []

    for i in range(len(values)):
        if not solver.best_solution_contains(i):
            continue

        packed_items.append(i)
        packed_values.append(values[i])
        packed_weights.append(weights[0][i])

    report_solution(
        packed_weights=packed_weights,
        packed_values=packed_values,
        packed_items=packed_items,
    )


def solve_cp_sat_single(instance: KnapsackInstance, config: KnapsackSolverConfig) -> KnapsackSolution:
    model = CpModel()
    df = pd.DataFrame(data={"weight": instance.weights, "value": instance.values})
    bs = model.new_bool_var_series("b", df.index)
    model.add(bs @ df["weight"] <= instance.capacity)
    model.maximize(bs @ df["value"])

    solver = CpSolver()
    solver.max_time_in_seconds = config.time_limit
    solver.parameters.relative_gap_limit = config.opt_tol
    status = solver.solve(model)

    if status not in (OPTIMAL, FEASIBLE):
        return KnapsackSolution(selected_items=[], objective=0, upper_bound=0)

    solution = KnapsackSolution(
        selected_items=[i for i in range(len(instance.values)) if solver.value(bs[i])],
        objective=solver.objective_value,
        upper_bound=solver.best_objective_bound
    )

    return solution


if __name__ == "__main__":
    values = [randint(1, 800) for _ in range(N_ITEMS)]
    weights = [randint(1, 100) for _ in range(N_ITEMS)]
    capacity = 850

    knap_instance = KnapsackInstance(values=values, weights=weights, capacity=capacity)
    solv_config = KnapsackSolverConfig()
    solution = solve_cp_sat_single(instance=knap_instance, config=solv_config)
    print("Chosen items: ", solution.selected_items)
    print("Objective: ", solution.objective)
    print("Upper bound: ", solution.upper_bound)
