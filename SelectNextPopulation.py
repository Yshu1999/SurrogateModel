import CrowdingDistance as cd
import numpy as np


class SelectNextPopulation:
    @staticmethod
    def select_next_population(combined_population, combined_fitness_values, fronts, pop_size):
        """Select the next generation population for NSGA-II."""
        next_population = []
        current_pop_size = 0
        front_index = 0

        while current_pop_size < pop_size and front_index < len(fronts):
            front = fronts[front_index]

            if current_pop_size + len(front) <= pop_size:
                next_population.extend(front)
                current_pop_size += len(front)
            else:
                # Calculate crowding distance and sort based on it
                distances = cd.CrowdingDistance.crowding_distance(combined_fitness_values, front)
                sorted_indices = np.argsort(-distances)
                sorted_front = [front[i] for i in sorted_indices]

                # Add individuals to fill the population
                remaining_slots = pop_size - current_pop_size
                next_population.extend(sorted_front[:remaining_slots])
                current_pop_size += remaining_slots

            front_index += 1

        # Create the final next population array
        next_population = combined_population[next_population]
        return next_population
