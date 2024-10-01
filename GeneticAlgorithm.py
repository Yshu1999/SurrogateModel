import concurrent

import numpy as np
import pandas as pd

import GeneratePopulation as gp
import NonDominatedSorting as nds
import FitnessEvaluator as fe
import SelectNextPopulation as snp
import matplotlib.pyplot as plt
import csv
import MLPClassifier as mlp

from DataProcessor import DatasetProcessor


class GeneticAlgorithm:
    def __init__(self, num_gen, problem, pop_size, param_size, lb, ub):
        self.num_gen = num_gen
        self.problem = problem
        self.pop_size = pop_size
        self.param_size = param_size
        self.lb = lb
        self.ub = ub
        self.pop = None
        self.pop_pure_ga = None
        self.pop_hybrid_ga = None
        self.fitness_values_hybrid_ga = None
        self.fitness_values_pure_ga = None
        self.best_fitness_values_pure_ga = []
        self.best_fitness_values_hybrid_ga = []

    def initialize(self):
        # Initialize the population
        self.pop = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                         self.ub).generate_population()
        self.pop_hybrid_ga = self.pop.copy()
        self.pop_pure_ga = self.pop.copy()

    def run(self):
        random_seed = 42
        random_state = np.random.RandomState(random_seed)
        # Evaluate the initial population's fitness
        self.fitness_values_pure_ga = fe.FitnessEvaluator(self.problem).evaluate_fitness(self.pop)
        self.fitness_values_hybrid_ga = self.fitness_values_pure_ga.copy()  # Same initial fitness values for hybrid GA

        # Record initial population
        self._record_generation(0, self.pop, self.fitness_values_pure_ga, self.pop, self.fitness_values_hybrid_ga)

        training_pop = self.pop
        training_fitness = self.fitness_values_hybrid_ga.reshape(-1, 1)

        # Run the genetic algorithm for num_gen generations
        for gen in range(1, self.num_gen + 1):
            # Generate offsprings for pure GA and hybrid GA separately
            offsprings_pure_ga = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                                       self.ub).generate_offsprings(
                self.pop_pure_ga, self.fitness_values_pure_ga, random_state)

            offsprings_hybrid_ga = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                                         self.ub).generate_offsprings(
                self.pop_hybrid_ga, self.fitness_values_hybrid_ga, random_state)

            # Parallel execution for fitness evaluation (pure GA and hybrid GA)
            offsprings_fitness_values_pure_ga, offsprings_fitness_values_hybrid = self.run_parallel_evaluations(
                gen, offsprings_pure_ga, offsprings_hybrid_ga, training_pop, training_fitness)

            # Combine population and offsprings for pure GA
            combined_pure_ga = np.vstack((self.pop_pure_ga, offsprings_pure_ga))
            combined_fitness_values_pure_ga = np.hstack(
                (self.fitness_values_pure_ga, offsprings_fitness_values_pure_ga))

            # Combine population and offsprings for hybrid GA
            combined_hybrid_ga = np.vstack((self.pop_hybrid_ga, offsprings_hybrid_ga))
            combined_fitness_values_hybrid_ga = np.hstack(
                (self.fitness_values_hybrid_ga, offsprings_fitness_values_hybrid))

            # Selection for pure GA
            self.pop_pure_ga, self.fitness_values_pure_ga = self.update_population_pure_ga(
                combined_pure_ga, combined_fitness_values_pure_ga, gen)

            # Selection for hybrid GA
            self.pop_hybrid_ga, self.fitness_values_hybrid_ga = self.update_population_hybrid_ga(
                combined_hybrid_ga, combined_fitness_values_hybrid_ga, gen)
            if gen == 2:
                training_pop = np.vstack((training_pop, self.pop_hybrid_ga))
                training_fitness = np.vstack((training_fitness, self.fitness_values_hybrid_ga.reshape(-1, 1)))

            # Record the best fitness and population for this generation (pure GA)
            self._record_generation(gen, self.pop_pure_ga, self.fitness_values_pure_ga, self.pop_hybrid_ga,
                                    self.fitness_values_hybrid_ga)

    def run_parallel_evaluations(self, gen, offsprings_pure_ga, offsprings_hybrid_ga, training_pop, training_fitness):
        # Define parallel evaluation for pure GA and hybrid GA
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_pure_ga = executor.submit(self.evaluate_pure_ga, offsprings_pure_ga)
            future_hybrid_ga = executor.submit(self.evaluate_hybrid_ga, gen, offsprings_hybrid_ga, training_pop,
                                               training_fitness)
            offsprings_fitness_values_pure_ga = future_pure_ga.result()
            offsprings_fitness_values_hybrid = future_hybrid_ga.result()

        return offsprings_fitness_values_pure_ga, offsprings_fitness_values_hybrid

    def evaluate_pure_ga(self, offsprings):
        # Pure GA fitness evaluation (applied on all generations)
        return fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)

    def evaluate_hybrid_ga(self, gen, offsprings, training_pop, training_fitness):
        # Hybrid GA fitness evaluation (predict on odd generations)
        if gen % 2 == 0:
            return fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)
        else:
            return self.predict_fitness(gen, training_pop, pd.DataFrame(training_fitness), offsprings)

    def update_population_pure_ga(self, combined, combined_fitness_values, gen):
        # Selection logic for pure GA
        if combined_fitness_values.ndim == 1:
            sorted_indices = np.argsort(combined_fitness_values)
            best_indices = sorted_indices[:min(self.pop_size, len(sorted_indices))]
            new_pop = combined[best_indices]
            new_fitness_values = combined_fitness_values[best_indices]
        else:
            fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)
            new_pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                        self.pop_size)

        return new_pop, new_fitness_values

    def update_population_hybrid_ga(self, combined, combined_fitness_values, gen):
        # Selection logic for hybrid GA (uses predicted fitness's in odd generations)
        if combined_fitness_values.ndim == 1:
            sorted_indices = np.argsort(combined_fitness_values)
            best_indices = sorted_indices[:min(self.pop_size, len(sorted_indices))]
            new_pop = combined[best_indices]
            new_fitness_values = combined_fitness_values[best_indices]
        else:
            fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)
            new_pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                        self.pop_size)

            # Hybrid fitness evaluation (in odd generations, predicted fitness is used)

        return new_pop, new_fitness_values

    def _record_generation(self, gen, population_pure_ga, fitness_values_pure_ga, population_hybrid_ga,
                           fitness_values_hybrid_ga):
        # Record results for pure GA
        with open(f'C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results_pure.csv', 'a', newline='') as f_pure:
            writer_pure = csv.writer(f_pure)
            best_fitness_value_pure = np.min(fitness_values_pure_ga)  # Assuming minimization for pure GA
            best_individual_pure = population_pure_ga[np.argmin(fitness_values_pure_ga)]
            self.best_fitness_values_pure_ga.append(best_fitness_value_pure)
            # Write to the CSV file for pure GA
            writer_pure.writerow([gen, best_fitness_value_pure] + list(best_individual_pure))
            # Print the result for pure GA
            print(
                f"Generation {gen} (Pure GA): Best Fitness Value = {best_fitness_value_pure}, Best Individual = {best_individual_pure}")

        # Record results for hybrid GA
        with open(f'C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results_hybrid.csv', 'a', newline='') as f_hybrid:
            writer_hybrid = csv.writer(f_hybrid)
            best_fitness_value_hybrid = np.min(fitness_values_hybrid_ga)  # Assuming minimization for hybrid GA
            best_individual_hybrid = population_hybrid_ga[np.argmin(fitness_values_hybrid_ga)]
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
        ax[0].legend()

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
        dp = DatasetProcessor(training_pop, training_fitness)
        mlp_model = mlp.MLPClassifierModel(dp)
        if gen == 1:
            mlp_model.train()
        else:
            mlp_model.retrain_model()
        offsprings_predicted_values = np.array([])
        for i in range(self.pop_size):
            current_predicted_values = []
            for j in range(len(training_pop)):
                current_predicted = mlp_model.predict_with_saved_model(
                    pd.DataFrame(np.concatenate((offsprings[i], training_pop[j]), axis=0).reshape(1, -1),
                                 columns=[f'X{j + 1}' for j in range(0, 2 * self.param_size)]))
                current_predicted_values.append(current_predicted[0])
            val = dp.forClassification(training_fitness, current_predicted_values)
            # Assuming val is the array you want to concatenate with offsprings_predicted_values
            offsprings_predicted_values = np.append(offsprings_predicted_values, val)

        return offsprings_predicted_values
