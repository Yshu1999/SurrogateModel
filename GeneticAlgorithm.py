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
        self.fitness_values = None
        self.best_fitness_values = []

    def initialize(self):
        # Initialize the population
        self.pop = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                         self.ub).generate_population()

    def run(self):
        random_seed = 42
        random_state = np.random.RandomState(random_seed)
        # Evaluate the initial population's fitness
        self.fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(self.pop)
        print(
            f"Population[{0}] :\n {pd.concat([pd.DataFrame(self.fitness_values, columns=['fitness']), pd.DataFrame(self.pop)], axis=1)}")

        # Clear the CSV file and write headers
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Generation', 'Best Fitness Value'] + [f'parameter {i + 1}' for i in
                                                              range(self.param_size)]
            writer.writerow(headers)

        # Write initial generation's best fitness and population
        self._record_generation(0)
        training_pop = self.pop
        training_fitness = self.fitness_values.reshape(-1, 1)

        # Run the genetic algorithm for num_gen generations
        for gen in range(1, self.num_gen + 1):
            offsprings = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                               self.ub).generate_offsprings(self.pop, self.fitness_values, random_state )

            if gen % 2 == 0:
                offsprings_fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)
            else:
                offsprings_fitness_values = self.predict_fitness(gen, training_pop, pd.DataFrame(training_fitness),
                                                                 offsprings)
            # Combine population and offsprings
            combined = np.vstack((self.pop, offsprings))
            combined_fitness_values = np.hstack((self.fitness_values, offsprings_fitness_values))

            if self.fitness_values.ndim == 1:
                sorted_indices = np.argsort(combined_fitness_values)
                best_indices = sorted_indices[:min(self.pop_size, len(sorted_indices))]

                self.pop = combined[best_indices]
                self.fitness_values = combined_fitness_values[best_indices]
            else:
                fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)
                self.pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                             self.pop_size)
                self.fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(self.pop)
            if gen % 2 == 0:
                training_pop = np.vstack((training_pop, self.pop))
                training_fitness = np.vstack((training_fitness, self.fitness_values.reshape(-1, 1)))
                print(len(training_pop))
            # Record the best fitness and population for this generation
            self._record_generation(gen)
            print(
                f"Population[{gen}] :\n {pd.concat([pd.DataFrame(self.fitness_values, columns=['fitness']), pd.DataFrame(self.pop)], axis=1)}")

    def _record_generation(self, gen):
        if self.fitness_values.ndim == 1:
            best_index = np.argmin(self.fitness_values)
        else:
            fronts, _ = nds.NonDominatedSorting().nds(self.fitness_values)
            best_index = fronts[0][0]

        best_fitness = self.fitness_values[best_index]
        best_population = self.pop[best_index]
        self.best_fitness_values.append(best_fitness)
        print(f"Best Fitness Value: {best_fitness} at Generation {gen}")
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fitness] + best_population.tolist())

    def stats(self):
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
