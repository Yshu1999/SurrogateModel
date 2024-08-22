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
        population = np.random.uniform(np.zeros(self.param_size), np.ones(self.param_size),
                                       (self.pop_size, self.param_size))
       # population = np.round(population, 2)
        return population

    def generate_offsprings(self, population, fitness_values):
        offsprings = []
        parents = []

        # Tournament Selection
        is_multi_objective = fitness_values.ndim > 1

        for j in range(2 * self.num_parents):
            tournament_selection_indexes = np.random.choice(len(population), 3, replace=False)
            print(tournament_selection_indexes)
            # Get fitness values for the selected individuals
            tournament_fitness = fitness_values[tournament_selection_indexes]
            print(tournament_fitness)

            if is_multi_objective:  # Multi-objective case
                selection_criteria = np.sum(tournament_fitness, axis=1)
            else:  # Single-objective case
                selection_criteria = tournament_fitness

            selected = tournament_selection_indexes[np.argmin(selection_criteria)]
            parents.append(population[selected])
        # Crossover
        for i in range(0, self.num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = self.sbx(parent1, parent2)
            offsprings.append(child1)
            offsprings.append(child2)

        return np.array(offsprings)

    def sbx(self, parent1, parent2):
        eta = 100 #0.95
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

        # Clip the values to be between 0 and 1
        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)

        # Apply polynomial mutation
        mutated_child1 = self.polynomial_mutation(child1[np.newaxis, :], mutation_rate=0.3)[0]
        mutated_child2 = self.polynomial_mutation(child2[np.newaxis, :], mutation_rate=0.3)[0]

        return mutated_child1, mutated_child2

    def polynomial_mutation(self, offspring, mutation_rate=0.05, eta_m=20):
        # Ensure offspring is a 2D array
        if offspring.ndim == 1:
            offspring = np.expand_dims(offspring, axis=0)

        num_individuals, num_variables = offspring.shape
        mutated_offspring = np.copy(offspring)

        for i in range(num_individuals):
            for j in range(num_variables):
                if np.random.rand() < mutation_rate:
                    y = offspring[i, j]
                    yl = 0  # Lower bound of the variable
                    yu = 1  # Upper bound of the variable
                    delta = 0

                    if y < 0.5:
                        delta = (2 * y + (1 - 2 * y) * np.random.rand()) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - y) + (2 * y - 1) * np.random.rand()) ** (1 / (eta_m + 1))

                    mutated_offspring[i, j] = np.clip(y + delta * (yu - yl), yl, yu)
                    #mutated_offspring[i, j] = y + delta * (yu - yl)

        return mutated_offspring
