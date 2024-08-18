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

    def geneticalgorithm(self):
        pop = gp.GeneratePopulation(self.pop_size, self.param_size, self.parents_size, self.lb,
                                    self.ub).generate_population()
        fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(pop)
        for gen in range(self.num_gen):
            print(f"Generation {gen + 1}")
            # Generate offsprings
            offsprings = gp.GeneratePopulation(self.pop_size, self.param_size, self.lb,
                                               self.ub, self.parents_size, ).generate_offsprings(pop, fitness_values)
            offsprings_fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(offsprings)
            # Combine population and offsprings
            combined = np.vstack((pop, offsprings))
            combined_fitness_values = np.vstack((fitness_values, offsprings_fitness_values))

            if fitness_values.ndim == 1:
                sorted_indices = np.argsort(combined_fitness_values)

                # Step 2: Select the best 50 (or fewer if you have fewer individuals) based on fitness
                best_indices = sorted_indices[:self.pop_size]
                # Step 3: Select the corresponding population members
                pop = combined[best_indices]
                fitness_values = combined_fitness_values[best_indices]
            else:
                # Perform non-dominated sorting
                fronts, ranks = nds.NonDominatedSorting().nds(combined_fitness_values)  #for multiobj

                # Select next generation population
                pop = snp.SelectNextPopulation().select_next_population(combined, combined_fitness_values, fronts,
                                                                        self.pop_size)
                fitness_values = fe.FitnessEvaluator(self.problem).evaluate_fitness(pop)

            # Return the final population and fitness values
        return pop, fitness_values
