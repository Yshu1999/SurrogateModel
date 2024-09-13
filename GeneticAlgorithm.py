import numpy as np
import GeneratePopulation as gp
import NonDominatedSorting as nds
import FitnessEvaluator as fe
import SelectNextPopulation as snp
import matplotlib.pyplot as plt
import csv



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
        # Evaluate the initial population's fitness
        self.fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(self.pop)

        # Clear the CSV file and write headers
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Generation', 'Best Fitness Value'] + [f'parameter {i + 1}' for i in
                                                              range(self.param_size)]
            writer.writerow(headers)

        # Write initial generation's best fitness and population
        self._record_generation(0)

        # Run the genetic algorithm for num_gen generations
        for gen in range(1, self.num_gen + 1):
            offsprings = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                               self.ub).generate_offsprings(self.pop, self.fitness_values)

            offsprings_fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)
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

            # Record the best fitness and population for this generation
            self._record_generation(gen)

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
        # Return or print the best solution after the GA run
        best_gen = np.argmin(self.best_fitness_values)
        best_fitness = self.best_fitness_values[best_gen]

        # Plotting the best fitness values against the generation numbers
        plt.figure(figsize=(10, 6))
        plt.plot(range(10), self.best_fitness_values[:10], marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value vs. Generation')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
