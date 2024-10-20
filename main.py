import numpy as np
from GeneticAlgorithm import GeneticAlgorithm


def main():
    # Define problem parameters
    num_gen = 10  # Number of generations
    pop_size = 10  # Population size
    param_size = 13  # Number of parameters (or genes)

    # Problem-specific setup, for example ZDT1 problem
    problem = "g01"

    # Create an instance of the GeneticAlgorithm
    ga = GeneticAlgorithm(num_gen, pop_size, param_size, problem)

    # Initialize the population
    ga.initialize()

    # Run the genetic algorithm
    ga.run()

    # Display the best solution at the end of the GA run
    ga.stats()

    # Optionally, retrieve the final population and fitness values
    final_population, final_fitness_values = ga.pop, ga.fitness_values
    print("Final Population:\n", final_population)
    print("Final Fitness Values:\n", final_fitness_values)


if __name__ == "__main__":
    main()
