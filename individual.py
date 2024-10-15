import numpy as np
class Individual:
    def __init__(self, solution):
        self.solution = solution
        self.fitness = None

    def evaluate_fitness(self, fitness_function, penalty_function=None):
        """Evaluates fitness with optional penalty."""
        solution_reshaped = self.solution if self.solution.ndim > 1 else self.solution.reshape(1, -1)
        self.fitness = fitness_function(solution_reshaped)
        return self.fitness

    def penalty_function(self, solution):
        """Calculates penalty based on constraint violations."""
        total_penalty = 0
        for constraint in constraints:
            violation = constraint(solution)
            if violation > 0:  # If the constraint is violated
                total_penalty += violation   # Apply quadratic penalty
        return total_penalty
