from ortools.sat.python.cp_model import (
    CpModel,
    CpSolver,
    OPTIMAL,
    Domain,
    FEASIBLE,
    CHOOSE_FIRST,
    SELECT_MIN_VALUE,
    FIXED_SEARCH,
)
import pandas as pd
from google.protobuf import text_format


def solve_simple_case():
    model = CpModel()
    solver = CpSolver()

    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")

    model.add(x + y <= 30)
    model.maximize(30 * x + 50 * y)

    status = solver.solve(model)

    assert status == OPTIMAL

    print(f"x={solver.value(x)},  y={solver.value(y)}")


def build_vars_from_df():
    model = CpModel()
    solver = CpSolver()
    idx = pd.Index(range(10), name="index")
    x = model.new_int_var_series("x", idx, 0, 100)

    df = pd.DataFrame(
        data={"weight": [1 for _ in range(10)], "value": [3 for _ in range(10)]},
        index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    bs = model.new_bool_var_series("b", df.index)
    model.Add(bs @ df["weight"] <= 100)
    model.Maximize(bs @ df["value"])

    status = solver.solve(model)
    print(f"bs={solver.values(bs)}")

    assert status == OPTIMAL


def use_domain_vars():
    model = CpModel()
    domain = Domain.from_values([2, 3, 17, 95])
    domain_2 = Domain.from_intervals([[1, 200], [30, 92]])

    domain_3 = domain.union_with(domain_2)

    x = model.NewIntVarFromDomain(domain_3, "x")

    solver = CpSolver()
    y = model.new_int_var(0, 100, "y")

    model.add(x + y < 1000)
    model.add(x + y > 10)
    model.maximize(3 * x + 12 * y)

    status = solver.solve(model)
    print(f"x = {solver.value(x)}, y = {solver.value(y)}")
    assert status == OPTIMAL


def lexicographic_optimization():
    model = CpModel()
    solver = CpSolver()
    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")

    model.add(x + y <= 30)
    model.Maximize(30 * x + 50 * y)
    solver.solve(model)
    model.add(30 * x + 50 * y == int(solver.objective_value))
    model.minimize(40 * x + y * 25)
    solver.solve(model)

    print(f"x = {solver.value(x)}, y = {solver.value(y)}")


def linear_constraints():
    model = CpModel()
    solver = CpSolver()

    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")
    z = model.new_int_var(0, 100, "z")

    model.add(10 * x + 15 * y <= 100)
    model.add(x + z == 2 * y)
    model.add(x + y != z)
    model.add(x <= z - 1)

    model.maximize(30 * y + 10 * x)

    solver.solve(model)
    print(f"x = {solver.value(x)}, y = {solver.value(y)}, z = {solver.value(z)}")


def logical_constraints():
    model = CpModel()
    solver = CpSolver()

    b1 = model.NewBoolVar("b1")
    b2 = model.NewBoolVar("b2")
    b3 = model.NewBoolVar("b3")

    model.add_bool_or(b1, b2, b3)
    model.AddBoolAnd(b1, ~b2, ~b3)
    model.AddBoolXOr(b1, b2, b3)
    model.add_implication(b1, b2)


def absolute_min_max():
    model = CpModel()

    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")
    z = model.new_int_var(0, 100, "z")

    abs_xyz = model.NewIntVar(0, 200, "|x+z|")
    model.AddAbsEquality(target=abs_xyz, expr=x + z)

    max_xyz = model.NewIntVar(0, 100, "max(x, y, z)")
    model.AddMaxEquality(max_xyz, [x, y, z])

    min_xyz = model.new_int_var(-100, 100, "min(x, y, z)")
    model.add_min_equality(min_xyz, [x, y, z])

    # more efficient linear constraint
    max_xyz = model.new_int_var(0, 100, "max_xyz")
    model.add(max_xyz >= x)
    model.add(max_xyz >= y)
    model.add(max_xyz >= z)

    model.minimize(max_xyz)


def multiply_divide_modulo():
    """
    multiplication is a nono in linear optimization, usually tools convert non linear problems to linear problem
    :return:
    """
    model = CpModel()
    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")
    z = model.new_int_var(0, 100, "z")
    xyz = model.NewIntVar(-100 * 100 * 100, 100**3, "x*y*z")
    model.AddMultiplicationEquality(xyz, [x, y, z])  # xyz = x*y*z
    model.AddModuloEquality(x, y, 3)  # x = y % 3
    model.AddDivisionEquality(x, y, z)  # x = y // z


def solve_tour():
    model = CpModel()
    solver = CpSolver()

    dgraph = {
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

    edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for u, v in dgraph.keys()}
    model.add_circuit([(u, v, var) for (u, v), var in edge_vars.items()])
    obj = sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items())

    model.minimize(obj)

    status = solver.solve(model)

    assert status in (OPTIMAL, FEASIBLE)

    tour = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x)]

    print("tour: ", tour)


def export_model(model: CpModel, filename: str):
    with open(filename, "w") as file:
        file.write(text_format.MessageToString(model.Proto()))


def import_model(filename: str) -> CpModel:
    model = CpModel()
    with open(filename, "r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model


def handle_assumptions(model: CpModel, b1, b2, b3):
    model.add_assumptions([b1, not b2])
    model.add_assumption(b3)
    model.clear_assumptions()


def hinting(model, x, y, solver):
    model.add_hint(x, 1)  # x will probably be 1
    model.add_hint(y, 2)  # y will probably be 2

    solver.parameters.debug_crash_on_bad_hint = True  # throw error if hints are wrong


def logging(solver: CpSolver):
    solver.parameters.log_search_progress = True
    solver.log_callback = print

    # note: primer includes a cp-sat log analyzer tool for nice viz of search progress


def decision_strategy(model: CpModel, x):
    model.add_decision_strategy([x], CHOOSE_FIRST, SELECT_MIN_VALUE)


if __name__ == "__main__":
    # build_vars_from_df()
    # solve_simple_case()
    # use_domain_vars()
    # lexicographic_optimization()
    # linear_constraints()
    solve_tour()
