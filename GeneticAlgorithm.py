import numpy as np
import pandas as pd

import GeneratePopulation as gp
import NonDominatedSorting as nds
import FitnessEvaluator as fe
import SelectNextPopulation as snp
import matplotlib.pyplot as plt
import csv
from individual import Individual


class GeneticAlgorithm:
    def __init__(self, num_gen, pop_size, param_size, lb, ub, problem):
        self.num_gen = num_gen
        self.problem = problem
        self.pop_size = pop_size
        self.param_size = param_size
        self.lb = lb
        self.ub = ub
        self.pop = None
        self.fitness_values = None
        self.best_fitness_values = []

    def initialize(self):
        # Initialize the population as a list of Individual objects
        self.pop = [Individual(np.random.uniform(self.lb, self.ub, self.param_size)) for _ in range(self.pop_size)]

    # In GeneticAlgorithm class, run method
    def run(self):
        # Evaluate the initial population's fitness (already compatible with Individual objects)
        self.evaluate_population_fitness(self.pop)

        # Clear the CSV file and write headers
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Generation', 'Best Fitness Value'] + [f'parameter {i + 1}' for i in range(self.param_size)]
            writer.writerow(headers)

        # Write initial generation's best fitness and population
        self._record_generation(0)

        # Run the genetic algorithm for num_gen generations
        for gen in range(1, self.num_gen + 1):

            offsprings = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb, self.ub).generate_offsprings(
                self.pop)

            self.evaluate_population_fitness(offsprings)

            # Combine parent and offspring populations
            combined = self.pop + offsprings

            # Combine fitness values for the parents and offsprings
            combined_fitness_values = np.array([individual.fitness for individual in combined])

            # If single-objective optimization
            if np.isscalar(self.pop[0].fitness):
                combined.sort(key=lambda ind: ind.fitness)

                # Select the best individuals based on fitness
                self.pop = combined[:self.pop_size]
                print(
                    f"Population[{gen}] :\n {pd.concat([pd.DataFrame(combined_fitness_values, columns=['fitness']), pd.DataFrame(combined)], axis=1)}")
            else:
                # For multi-objective optimization, use non-dominated sorting and selection
                fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)
                self.pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                             self.pop_size)
                self.evaluate_population_fitness(self.pop)

            # Record the best fitness and population for this generation
            self._record_generation(gen)

    def evaluate_population_fitness(self, pop):
        """Evaluate fitness for each individual in the population."""
        fitness_evaluator = fe.FitnessEvaluator(self.problem, self.param_size)  # Create a fitness evaluator

        # Loop through each individual and evaluate its fitness
        for individual in pop:
            individual.evaluate_fitness(fitness_evaluator.evaluate_fitness,
                                        individual.penalty_function)  # Call evaluate_fitness for each individual

    def _record_generation(self, gen):
        # Find the individual with the best fitness (assuming minimization)
        best_individual = min(self.pop, key=lambda ind: ind.fitness)

        best_fitness = best_individual.fitness
        best_solution = best_individual.solution  # Access the solution of the best individual

        self.best_fitness_values.append(best_fitness)

        print(f"Best Fitness Value: {best_fitness} at Generation {gen}")

        # Write to the CSV file
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fitness] + best_solution.tolist())  # Convert solution to a list if needed

    def stats(self):
        # Return or print the best solution after the GA run
        best_gen = np.argmin(self.best_fitness_values)
        best_fitness = self.best_fitness_values[best_gen]

        # Plotting the best fitness values against the generation numbers
        plt.figure(figsize=(self.num_gen, 12))
        plt.plot(range(self.num_gen), self.best_fitness_values[:self.num_gen], marker='o', linestyle='-', color='b')
        plt.xticks(np.arange(0, self.num_gen, 1))
        plt.title('Best Fitness Value vs. Generation')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
