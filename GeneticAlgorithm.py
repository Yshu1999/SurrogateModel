import concurrent

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
    def __init__(self, num_gen, pop_size, param_size, problem):
        self.num_gen = num_gen
        self.problem = problem
        self.pop_size = pop_size
        self.param_size = param_size
        self.fitness_evaluator = fe.FitnessEvaluator(self.problem, self.param_size)
        self.lb, self.ub = self.fitness_evaluator.get_bounds()
        self.pop = None
        self.fitness_values = None
        self.best_fitness_values = []
        self.training_pop = []
        self.training_fitness = []
        self.pop_pure_ga = None
        self.pop_hybrid_ga = None
        self.fitness_values_hybrid_ga = None
        self.fitness_values_pure_ga = None
        self.best_fitness_values_pure_ga = []
        self.best_fitness_values_hybrid_ga = []

    def initialize(self):
        # Initialize the population as a list of Individual objects
        self.pop = [Individual(np.random.uniform(self.lb, self.ub, self.param_size)) for _ in range(self.pop_size)]
        self.evaluate_population_fitness(self.pop)
        for individual in self.pop:
            self.training_pop.append(individual.solution)  # Append the solution (parameter values)
            self.training_fitness.append(individual.fitness)  # Append the fitness value
        for idx, individual in enumerate(self.pop):
            print(f"Individual[{idx}]: Solution = {individual.solution}, Fitness = {individual.fitness}")
        # Initialize the population
        self.pop_hybrid_ga = self.pop.copy()
        self.pop_pure_ga = self.pop.copy()

    def run(self):
        random_seed = 42
        random_state = np.random.RandomState(random_seed)
        # Record initial population
        self._record_generation(0, self.pop, self.fitness_values_pure_ga, self.pop, self.fitness_values_hybrid_ga)

        # Run the genetic algorithm for num_gen generations
        for gen in range(1, self.num_gen + 1):
            # Generate offsprings for pure GA and hybrid GA separately
            # Initialize GeneratePopulation once
            x = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb, self.ub)
            # Generate offsprings for pure GA
            offsprings_pure_ga = x.generate_offsprings(
                self.pop_pure_ga, random_state
            )
            # Generate offsprings for hybrid GA
            offsprings_hybrid_ga = x.generate_offsprings(
                self.pop_hybrid_ga, random_state
            )
            self.run_parallel_evaluations(
                gen, offsprings_pure_ga, offsprings_hybrid_ga, self.training_pop, self.training_fitness)

            # Combine population and offsprings for pure GA
            combined_pure_ga = self.pop_pure_ga + offsprings_pure_ga
            combined_fitness_values_pure_ga = np.array([individual.fitness for individual in combined_pure_ga])
            # Combine population and offsprings for hybrid GA
            combined_hybrid_ga = self.pop_hybrid_ga + offsprings_hybrid_ga
            combined_fitness_values_hybrid_ga = np.array([individual.fitness for individual in combined_hybrid_ga])
            # Selection for pure GA
            self.pop_pure_ga = self.update_population(combined_pure_ga)
            # Selection for hybrid GA
            self.pop_hybrid_ga = self.update_population(combined_hybrid_ga)
            if gen % 2 == 0:
                for individual in self.pop_hybrid_ga:
                    self.training_pop.append(individual.solution)  # Append the solution (parameter values)
                    self.training_fitness.append(individual.fitness)  # Append the fitness value
            self._record_generation(gen, self.pop_pure_ga, self.fitness_values_pure_ga, self.pop_hybrid_ga,
                                    self.fitness_values_hybrid_ga)

    def run_parallel_evaluations(self, gen, offsprings_pure_ga, offsprings_hybrid_ga, training_pop, training_fitness):
        # Define parallel evaluation for pure GA and hybrid GA
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.evaluate_pure_ga, offsprings_pure_ga)
            executor.submit(self.evaluate_hybrid_ga, gen, offsprings_hybrid_ga, self.training_pop,
                            self.training_fitness)
        return

    def evaluate_pure_ga(self, offsprings):
        self.evaluate_population_fitness(offsprings)

    def evaluate_hybrid_ga(self, gen, offsprings, training_pop, training_fitness):
        # Hybrid GA fitness evaluation (predict on odd generations)
        if gen % 2 == 0:
            self.evaluate_population_fitness(offsprings)
        else:
            self.predict_fitness(gen, self.training_pop, pd.DataFrame(
                self.training_fitness), offsprings)


    def evaluate_population_fitness(self, pop):
        """Evaluate fitness for each individual in the population."""
        # Loop through each individual and evaluate its fitness
        for individual in pop:
            individual.evaluate_fitness(self.fitness_evaluator.evaluate_fitness)  # Call evaluate_fitness for each individual

    def update_population(self, combined):
        combined.sort(key=lambda ind: ind.fitness)
        # Select the best individuals based on fitness
        new_pop = combined[:self.pop_size]
        return new_pop

    def _record_generation(self, gen, population_pure_ga, fitnenew_popss_values_pure_ga, population_hybrid_ga,
                           fitness_values_hybrid_ga):
        # Record results for pure GA
        with open(f'C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results_pure.csv', 'a', newline='') as f_pure:
            writer_pure = csv.writer(f_pure)
            best_individual = min(self.pop_pure_ga, key=lambda ind: ind.fitness)
            best_fitness_value_pure = best_individual.fitness
            best_individual_pure = best_individual.solution  # Access the solution of the best individual
            self.best_fitness_values_pure_ga.append(best_fitness_value_pure)
            # Write to the CSV file for pure GA
            writer_pure.writerow([gen, best_fitness_value_pure] + list(best_individual_pure))
            # Print the result for pure GA
            print(
                f"Generation {gen} (Pure GA): Best Fitness Value = {best_fitness_value_pure}, Best Individual = {best_individual_pure}")

        # Record results for hybrid GA
        with open(f'C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results_hybrid.csv', 'a', newline='') as f_hybrid:
            writer_hybrid = csv.writer(f_hybrid)
            best_individual = min(self.pop_hybrid_ga, key=lambda ind: ind.fitness)
            best_fitness_value_hybrid = best_individual.fitness  # Assuming minimization for hybrid GA
            best_individual_hybrid = best_individual.solution
            self.best_fitness_values_hybrid_ga.append(best_fitness_value_hybrid)
            # Write to the CSV file for hybrid GA
            writer_hybrid.writerow([gen, best_fitness_value_hybrid] + list(best_individual_hybrid))
            # Print the result for hybrid GA
            print(
                f"Generation {gen} (Hybrid GA): Best Fitness Value = {best_fitness_value_hybrid}, Best Individual = {best_individual_hybrid}")

    def stats(self):
        best_gen_pure = np.argmin(self.best_fitness_values_pure_ga)
        best_fitness_pure = self.best_fitness_values_pure_ga[best_gen_pure]

        best_gen_hybrid = np.argmin(self.best_fitness_values_hybrid_ga)
        best_fitness_hybrid = self.best_fitness_values_hybrid_ga[best_gen_hybrid]

        # Assuming mlp.MLPClassifierModel.plot_error_vs_generation() returns error_list
        error_list = mlp.MLPClassifierModel.plot_error_vs_generation()
        generations = list(range(1, len(error_list) + 1))  # Generations: 1, 2, 3, ..., N

        # Create subplots: 2 rows, 1 column
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Best Fitness Value vs. Generation (for both pure and hybrid GA)
        ax[0].plot(range(self.num_gen), self.best_fitness_values_pure_ga[:self.num_gen], marker='o', linestyle='-',
                   color='b', label='Pure GA')
        ax[0].plot(range(self.num_gen), self.best_fitness_values_hybrid_ga[:self.num_gen], marker='s', linestyle='--',
                   color='g', label='Hybrid GA')
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
            offspring.fitness = val.item()  # Now it’s a 1D ndarray

        return  # Y No need to return anything, as offsprings are updated in-place
