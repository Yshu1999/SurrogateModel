import numpy as np
class Individual:
    def __init__(self, solution):
        self.solution = solution
        self.fitness = None

    def evaluate_fitness(self, fitness_function):
        """Evaluates fitness with optional penalty."""
        solution_reshaped = self.solution if self.solution.ndim > 1 else self.solution.reshape(1, -1)
        self.fitness = fitness_function(solution_reshaped)
        return self.fitness
