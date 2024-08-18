import numpy as np
import FitnessEvaluator as fe
import random


class GeneratePopulation:
    def __init__(self, pop_size, param_size,
                 lower_bound, upper_bound, num_parents):
        self.param_size = param_size
        self.pop_size = pop_size
        self.num_parents = num_parents
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate_population(self):
        population = np.random.uniform(self.lower_bound, self.upper_bound,
                                       (self.pop_size, self.param_size))
        return population

    def sbx(self, parent1, parent2):
        eta = 10
        child1 = np.empty(parent1.shape)
        child2 = np.empty(parent2.shape)
        for i in range(len(parent1)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        return child1, child2

    def generate_offsprings(self, population, fitness_values):
        offsprings = []
        parents = []

        # Tournament Selection
        for i in range(self.num_parents):
            tournament_selection_indexes = np.random.choice(len(population), 3, replace=False)
            if fitness_values.ndim > 1:  # Multi-objective case
                tournament_fitness = np.sum(fitness_values[tournament_selection_indexes], axis=1)
            else:  # Single-objective case
                tournament_fitness = fitness_values[tournament_selection_indexes]
            selected = tournament_selection_indexes[np.argmin(tournament_fitness)]
            parents.append(population[selected])

        # Crossover
        for i in range(2*self.num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = self.sbx(parent1, parent2)
            offsprings.append(child1)
            offsprings.append(child2)

        return np.array(offsprings)
