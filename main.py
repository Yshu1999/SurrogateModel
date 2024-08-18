import numpy as np
from GeneticAlgorithm import GeneticAlgorithm


def main():
    # Define problem parameters
    num_gen = 100  # Number of generations
    pop_size = 50  # Population size
    param_size = 30  # Number of parameters (or genes)
    lb = 0  # Lower bounds of the parameters
    ub = 1  # Upper bounds of the parameters
    parents_size = 50  # Number of parents

    # Problem-specific setup, for example ZDT1 problem
    problem = "ackley"

    # Create an instance of the GeneticAlgorithm
    ga = GeneticAlgorithm(num_gen, problem, pop_size, param_size, lb, ub, parents_size)

    # Run the genetic algorithm
    final_population, final_fitness_values = ga.geneticalgorithm()

    # Output or analyze the final results
    print("Final Population:\n", final_population)
    print("Final Fitness Values:\n", final_fitness_values)


if __name__ == "__main__":
    main()
