from pydantic import BaseModel, NonNegativeInt
from typing import Dict
from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE


class TSPInstance(BaseModel):
    distances: Dict[tuple, NonNegativeInt]


def solve_tsp_cpsat(instance: TSPInstance):
    model = CpModel()
    solver = CpSolver()

    edge_vars = {
        (i, j): model.new_bool_var(f"e_{i}_{j}") for i, j in instance.distances.keys()
    }

    circuit = list()
    obj_vals = list()
    for (i, j), var in edge_vars.items():
        circuit.append((i, j, var))
        obj_vals.append(instance.distances[i, j] * var)

    obj = sum(obj_vals)
    model.add_circuit(circuit)

    model.minimize(obj)
    status = solver.solve(model)

    assert status in (OPTIMAL, FEASIBLE)

    print([(i, j) for (i, j), var in edge_vars.items() if solver.value(var)])


if __name__ == "__main__":
    solve_tsp_cpsat(
        TSPInstance(
            distances={
                (0, 1): 13,
                (1, 0): 17,
                (1, 2): 16,
                (2, 1): 19,
                (0, 2): 22,
                (2, 0): 14,
                (3, 0): 15,
                (3, 1): 28,
                (3, 2): 25,
                (0, 3): 24,
                (1, 3): 11,
                (2, 3): 27,
            }
        )
    )
