import none
import numpy as np
import FitnessEvaluator as fe
import random


class GeneratePopulation:
    def __init__(self, pop_size, param_size,
                 lower_bound, upper_bound, ):
        self.param_size = param_size
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate_population(self):
        population = np.random.uniform(np.zeros(self.param_size), np.ones(self.param_size),
                                       (self.pop_size, self.param_size))
        return population

    def generate_offsprings(self, population, fitness_values, random_state):
        offsprings = []
        parents = []

        # Tournament Selection
        is_multi_objective = fitness_values.ndim > 1

        for j in range(2 * self.pop_size):
            tournament_selection_indexes = np.random.choice(len(population), 3, replace=False)
            # Get fitness values for the selected individuals
            #print(f"fitness_values size: {len(fitness_values)}")
            # print(f"tournament_selection_indexes: {tournament_selection_indexes}")
            tournament_fitness = fitness_values[tournament_selection_indexes]

            if is_multi_objective:  # Multi-objective case
                selection_criteria = np.sum(tournament_fitness, axis=1)
            else:  # Single-objective case
                selection_criteria = tournament_fitness

            selected = tournament_selection_indexes[np.argmin(selection_criteria)]
            parents.append(population[selected])
        # Crossover
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = self.sbx(parent1, parent2, random_state )
            offsprings.append(child1)
            offsprings.append(child2)

        return np.array(offsprings)[0:self.pop_size]

    def sbx(self, parent1, parent2, random_state=none, crossover_prob=0.9):
        eta = 20
        child1 = np.empty(parent1.shape)
        child2 = np.empty(parent2.shape)

        # Generate a random number to decide if crossover should happen
        if random_state.rand() < crossover_prob:
            for i in range(len(parent1)):
                u = random_state.rand()
                print("u", u)
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        else:
            # If no crossover, children are copies of parents
            child1 = parent1.copy()
            child2 = parent2.copy()

        # Apply polynomial mutation
        mutated_child1 = self.polynomial_mutation(child1[np.newaxis, :],random_state )[0]
        mutated_child2 = self.polynomial_mutation(child2[np.newaxis, :], random_state)[0]

        return mutated_child1, mutated_child2

    def polynomial_mutation(self, offspring, random_state = none, mutation_rate=0.05, eta_m=20):
        num_variables = offspring.shape[0]
        mutated_offspring = np.copy(offspring)

        for j in range(num_variables):
            if random_state.rand() < mutation_rate:
                y = offspring[j]
                yl = 0  # Lower bound of the variable
                yu = 1  # Upper bound of the variable

                u = random_state.rand()  # Random number between 0 and 1
                if u <= 0.5:
                    delta_i = (2 * u + (1 - 2 * u) * (1 - mutation_rate) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
                    delta = delta_i * (y - yl)
                else:
                    delta_i = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - mutation_rate) ** (eta_m + 1)) ** (
                            1 / (eta_m + 1))
                    delta = delta_i * (yu - y)

                # Apply the mutation and clip the value to ensure it stays within bounds
                mutated_offspring[j] = np.clip(y + delta, yl, yu)

        return mutated_offspring
