import numpy as np
import pandas as pd

import GeneratePopulation as gp
import NonDominatedSorting as nds
import FitnessEvaluator as fe
import SelectNextPopulation as snp
import matplotlib.pyplot as plt
import csv
from individual import Individual
import MLPClassifier as mlp

from DataProcessor import DatasetProcessor


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
        self.training_pop =[]
        self.training_fitness = []
    def initialize(self):
        # Initialize the population as a list of Individual objects
        self.pop = [Individual(np.random.uniform(self.lb, self.ub, self.param_size)) for _ in range(self.pop_size)]
        self.evaluate_population_fitness(self.pop)
        for individual in self.pop:
            self.training_pop.append(individual.solution)  # Append the solution (parameter values)
            self.training_fitness.append(individual.fitness)  # Append the fitness value
        for idx, individual in enumerate(self.pop):
            print(f"Individual[{idx}]: Solution = {individual.solution}, Fitness = {individual.fitness}")
    def run(self):
        # Evaluate the initial population's fitness (already compatible with Individual objects)
        #random_seed = 42
        #random_state = np.random.RandomState(random_seed)
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
            if gen % 2 == 0:
                self.evaluate_population_fitness(offsprings)
            else:
                self.predict_fitness(gen, self.training_pop, pd.DataFrame(
                    self.training_fitness), offsprings)

            # Combine parent and offspring populations
            combined = self.pop + offsprings
            # Combine fitness values for the parents and offsprings
            combined_fitness_values = np.array([individual.fitness for individual in combined])

            # If single-objective optimization
            if np.isscalar(self.pop[0].fitness):
                combined.sort(key=lambda ind: ind.fitness)
                # Select the best individuals based on fitness
                self.pop = combined[:self.pop_size]
                for idx, individual in enumerate(combined):
                    print(f"Individual[{idx}]: Solution = {individual.solution}, Fitness = {individual.fitness}")
            else:
                # For multi-objective optimization, use non-dominated sorting and selection
                fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)
                self.pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                             self.pop_size)
                self.fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(self.pop)
            if gen % 2 == 0:
                for individual in self.pop:
                    self.training_pop.append(individual.solution)  # Append the solution (parameter values)
                    self.training_fitness.append(individual.fitness)  # Append the fitness value
                print(len(self.training_pop))

            # Record the best fitness and population for this generation
            self._record_generation(gen)

    def evaluate_population_fitness(self, pop):
        """Evaluate fitness for each individual in the population."""
        fitness_evaluator = fe.FitnessEvaluator(self.problem, self.param_size)  # Create a fitness evaluator

        # Loop through each individual and evaluate its fitness
        for individual in pop:
            individual.evaluate_fitness(fitness_evaluator.evaluate_fitness)  # Call evaluate_fitness for each individual

    def _record_generation(self, gen):
        # Find the individual with the best fitness (assuming minimization)
        best_individual = min(self.pop, key=lambda ind: ind.fitness)

        best_fitness = best_individual.fitness
        best_solution = best_individual.solution  # Access the solution of the best individual

        self.best_fitness_values.append(best_fitness)

        print(f"Best Fitness Value: {best_fitness} at Generation {gen}")
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fitness] + best_solution.tolist())

    def stats(self):
        # Return or print the best solution after the GA run
        best_gen = np.argmin(self.best_fitness_values)
        best_fitness = self.best_fitness_values[best_gen]

        # Assuming mlp.MLPClassifierModel.plot_error_vs_generation() returns error_list
        error_list = mlp.MLPClassifierModel.plot_error_vs_generation()
        generations = list(range(1, len(error_list) + 1))  # Generations: 1, 2, 3, ..., N

        # Create subplots: 2 rows, 1 column
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Best Fitness Value vs. Generation
        ax[0].plot(range(self.num_gen), self.best_fitness_values[:self.num_gen], marker='o', linestyle='-', color='b')
        ax[0].set_xticks(np.arange(0, self.num_gen, 1))
        ax[0].set_title('Best Fitness Value vs. Generation')
        ax[0].set_xlabel('Generation')
        ax[0].set_ylabel('Best Fitness Value')
        ax[0].grid(True)

        # Plot 2: MSE vs. Generation
        ax[1].plot(generations, error_list, marker='o', linestyle='-', color='r', label='MSE')
        ax[1].set_title('MSE vs. Generation')
        ax[1].set_xlabel('Generation')
        ax[1].set_ylabel('Mean Squared Error (MSE)')
        ax[1].grid(True)
        ax[1].legend()

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Display the plots in one window
        plt.show()

    def predict_fitness(self, gen, training_pop, training_fitness, offsprings):
        dp = DatasetProcessor(training_pop, np.array(training_fitness).reshape(-1, 1))
        mlp_model = mlp.MLPClassifierModel(dp)

        # Train or retrain the model based on the generation number
        if gen == 1:
            mlp_model.train()
        else:
            mlp_model.retrain_model()

        # Loop through each offspring and update its fitness directly
        for i, offspring in enumerate(offsprings):
            current_predicted_values = []

            # Predict fitness for the offspring based on the training population
            for j in range(len(training_pop)):
                current_predicted = mlp_model.predict_with_saved_model(
                    pd.DataFrame(np.concatenate((offspring.solution, training_pop[j]), axis=0).reshape(1, -1),
                                 columns=[f'X{j + 1}' for j in range(0, 2 * self.param_size)]))
                current_predicted_values.append(current_predicted[0])

            # Calculate the final fitness value for the offspring
            val = dp.forClassification(training_fitness, current_predicted_values)

            # Convert the scalar value to ndarray before assigning it
            offspring.fitness = val.item() # Now it’s a 1D ndarray

        return  # Y No need to return anything, as offsprings are updated in-place




