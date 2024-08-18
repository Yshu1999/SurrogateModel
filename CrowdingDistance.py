import numpy as np


class CrowdingDistance:
    def crowding_distance(fitness_values, front):
        """Calculates the crowding distance for a given front.

    Args:
        fitness_values (ndarray): A 2D array where each row represents the fitness values of an individual.
        front (list): A list of indices representing the individuals in the front.

    Returns:
        ndarray: An array of crowding distances for the individuals in the front.
    """
        num_objectives = fitness_values.shape[1]
        distances = np.zeros(len(front))

        # For each objective, sort the individuals based on the objective value
        for i in range(num_objectives):
            sorted_indices = np.argsort(fitness_values[front, i])
            max_value = fitness_values[front[sorted_indices[-1]], i]
            min_value = fitness_values[front[sorted_indices[0]], i]

            # Assign infinity to boundary points
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf

            # Calculate distance for the rest of the points
            for j in range(1, len(front) - 1):
                distances[sorted_indices[j]] += (fitness_values[front[sorted_indices[j + 1]], i] -
                                                 fitness_values[front[sorted_indices[j - 1]], i]) / (
                                                        max_value - min_value + 1e-9)

        return distances

# Example usage:
# fitness_values = np.array([[0.1, 0.8], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
# front = [0, 1, 2, 3, 4]
# distances = crowding_distance(fitness_values, front)
