import none
import numpy as np
from individual import Individual
import FitnessEvaluator as fe
import random

class GeneratePopulation:
    def __init__(self, pop_size, param_size,
                 lower_bounds, upper_bounds, ):
        self.param_size = param_size
        self.pop_size = pop_size
        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds

    def generate_offsprings(self, population, random_state):
        offsprings = []
        parents = []
        for j in range(2 * self.pop_size):
            # Select 3 random individuals for tournament selection
            tournament_selection_indexes = np.random.choice(len(population), 3, replace=False)
            tournament_individuals = [population[idx] for idx in tournament_selection_indexes]
            # Get fitness values directly from the individual objects
            tournament_fitness = [ind.fitness for ind in tournament_individuals]
            selection_criteria = tournament_fitness  # Single-objective case
            # Select the individual with the best fitness
            selected_idx = np.argmin(selection_criteria)
            parents.append(tournament_individuals[selected_idx])
        # Crossover to produce offsprings
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i].solution  # Extract solution from parent
            parent2 = parents[i + 1].solution  # Extract solution from parent
            child1_solution, child2_solution = self.sbx(parent1, parent2, random_state)

            # Create new Individual objects for the offspring
            child1 = Individual(child1_solution)  # Assuming Individual class takes solution as input
            child2 = Individual(child2_solution)  # Assuming Individual class takes solution as input

            offsprings.append(child1)
            offsprings.append(child2)

        return offsprings

    def sbx(self, parent1, parent2, random_state=none, crossover_prob=0.9):
        eta = 20
        child1 = np.empty(parent1.shape)
        child2 = np.empty(parent2.shape)

        # Generate a random number to decide if crossover should happen
        if random_state.rand() < crossover_prob:
            for i in range(len(parent1)):
                u = random_state.rand()
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
                yl = self.lower_bound  # Lower bound of the variable
                yu = self.upper_bound  # Upper bound of the variable

                u = random_state.rand()  # Random number between 0 and 1
                if u <= 0.5:
                    delta_i = (2 * u + (1 - 2 * u) * (1 - mutation_rate) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
                    delta = delta_i * (y - yl)
                else:
                    delta_i = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - mutation_rate) ** (eta_m + 1)) ** (
                            1 / (eta_m + 1))
                    delta = delta_i * (yu - y)

                # Apply the mutation and clip the value to ensure it stays within bounds
                mutated_offspring[j] = np.clip(y + delta, yl[j], yu[j])

        return mutated_offspring
