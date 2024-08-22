import numpy as np
import GeneratePopulation as gp
import NonDominatedSorting as nds
import FitnessEvaluator as fe
import SelectNextPopulation as snp


class GeneticAlgorithm:
    def __init__(self, num_gen, problem, pop_size, param_size, lb, ub, parents_size):
        self.num_gen = num_gen
        self.problem = problem
        self.pop_size = pop_size
        self.param_size = param_size
        self.lb = lb
        self.ub = ub
        self.parents_size = parents_size

    import numpy as np
    def geneticalgorithm(self):
        pop = gp.GeneratePopulation(self.pop_size, self.param_size, self.parents_size, self.lb,
                                    self.ub).generate_population()
        fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(pop)
#write the best in gen 0 as well
        # Open a file in append mode
        with open('C:/Users/vyshn/OneDrive/Desktop/genetic_algorithm_results.txt', 'a') as f:
            for gen in range(self.num_gen):
                print(f"Generation {gen + 1}")

                # Generate offsprings
                offsprings = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                                   self.ub, self.parents_size).generate_offsprings(pop, fitness_values)
                #offsprings = np.round(offsprings, 2)
                offsprings_fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)
                # Combine population and offsprings
                combined = np.vstack((pop, offsprings))
                combined_fitness_values = np.hstack((fitness_values, offsprings_fitness_values))

                if fitness_values.ndim == 1:
                    sorted_indices = np.argsort(combined_fitness_values)

                    # Ensure not to access out-of-bound indices
                    best_indices = sorted_indices[:min(self.pop_size, len(sorted_indices))]

                    # Select the best individuals
                    pop = combined[best_indices]
                    fitness_values = combined_fitness_values[best_indices]
                    best_index = best_indices[0]
                else:
                    # Perform non-dominated sorting (for multi-objective case)
                    fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)

                    # Select next generation population
                    pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                            self.pop_size)
                    fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(pop)

                # Write the population, mutated offspring, and fitness values to the file
                np.savetxt(f, combined[best_index:best_index + 1], delimiter=',',
                           header=f"Generation {gen + 1} Best Population",
                           comments='')
                np.savetxt(f, combined_fitness_values[best_index:best_index + 1], delimiter=',',
                           header=f"Generation {gen + 1}  Best Fitness Values", comments='')

        # Return the final population and fitness values
        return pop, fitness_values
