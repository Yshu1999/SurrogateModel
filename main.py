import numpy as np
from GeneticAlgorithm import GeneticAlgorithm


def main():
    # Define problem parameters
    num_gen = 10  # Number of generations
    pop_size = 10 # Population size
    param_size = 10  # Number of parameters (or genes)
    lb = 0  # Lower bounds of the parameters
    ub = 1  # Upper bounds of the parameters

    # Problem-specific setup, for example ZDT1 problem
    problem = "ackley"

    # Create an instance of the GeneticAlgorithm
    ga = GeneticAlgorithm(num_gen, problem, pop_size, param_size, lb, ub)

    # Initialize the population
    ga.initialize()

    # Run the genetic algorithm
    ga.run()

    # Display the best solution at the end of the GA run
    ga.stats()


if __name__ == "__main__":
    main()
