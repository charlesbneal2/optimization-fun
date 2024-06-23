from pydantic import BaseModel, PositiveInt
from typing import List
from ortools.sat.python.cp_model import (
    CpModel,
    CpSolver,
    CpSolverSolutionCallback,
    IntVar,
)


class NQueensInstance(BaseModel):
    n_queens: PositiveInt


class NQueensSolution(BaseModel):
    solutions: List[List[int]]
    n_queens: PositiveInt

    @property
    def readable(self):
        res = []

        for solution in self.solutions:
            sol = []
            for placement in solution:
                rep = ["."] * self.n_queens
                rep[placement] = "Q"
                sol.append("".join(rep))
            res.append(sol)

        return res


class NQueensCallback(CpSolverSolutionCallback):
    def __init__(self, queens: List[IntVar]):
        super().__init__()
        self.__queens = queens
        self.__solutions = []

    @property
    def solutions(self) -> List[List[int]]:
        return self.__solutions

    def on_solution_callback(self):
        self.solutions.append(
            [self.value(self.__queens[i]) for i in range(len(self.__queens))]
        )

    def to_solution(self):
        return NQueensSolution(solutions=self.solutions, n_queens=len(self.__queens))


def cpsat_solve_n_queens(instance: NQueensInstance):
    model = CpModel()

    queens = [
        model.new_int_var(0, instance.n_queens - 1, f"x_{i}")
        for i in range(instance.n_queens)
    ]

    model.add_all_different(queens)

    model.add_all_different(queens[i] + i for i in range(instance.n_queens))
    model.add_all_different(queens[i] - i for i in range(instance.n_queens))

    callback = NQueensCallback(queens=queens)
    solver = CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.solve(model, callback)

    return callback.to_solution()


if __name__ == "__main__":
    instance = NQueensInstance(n_queens=10)
    solution = cpsat_solve_n_queens(instance)
    print("N solutions: ", len(solution.solutions))
    print(solution.readable[0])
