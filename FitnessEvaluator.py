import numpy as np


class FitnessEvaluator:
    def __init__(self, problem_name, num_variables=30):
        """
        Initialize the problem evaluator with a specific function.

        :param problem_name: String, specifies the function ("ackley", "bukin", "zdt1", etc.)
        :param num_variables: Number of decision variables (n). Default is 30.
        """
        self.num_variables = num_variables

        # Dictionary mapping problem names to methods
        self.problems = {
            "ackley": self.ackley,
            "bukin": self.bukin,
            "zdt1": self.zdt1,
            "zdt2": self.zdt2,

            "cross_in_tray": self.cross_in_tray,
            "drop_wave": self.drop_wave,
            "eggholder": self.eggholder,
            "gramacy_lee": self.gramacy_lee,
            "griewank": self.griewank,
            "holder_table": self.holder_table,
            "langermann": self.langermann,
            "levy": self.levy,
            "levy_n13": self.levy_n13,
            "rastrigin": self.rastrigin,
            "schaffer_n2": self.schaffer_n2,
            "schaffer_n4": self.schaffer_n4,
            "schwefel": self.schwefel,
            "shubert": self.shubert,
            "bohachevsky": self.bohachevsky,
            "perm_function": self.perm_function,
            "rotated_hyper_ellipsoid": self.rotated_hyper_ellipsoid,
            "sphere_function": self.sphere_function,
            "sum_of_different_powers": self.sum_of_different_powers,
            "sum_of_squares": self.sum_of_squares,
            "trid_function": self.trid_function,
            "booth_function": self.booth_function,
            "matyas_function": self.matyas_function,
            "mccormick_function": self.mccormick_function,
            "power_sum_function": self.power_sum_function,
            "zakharov_function": self.zakharov_function,
            "three_hump_camel_function": self.three_hump_camel_function,
            "six_hump_camel_function": self.six_hump_camel_function,
            "dixon_price_function": self.dixon_price_function,
            "rosenbrock_function": self.rosenbrock_function,
            "de_jong_function_5": self.de_jong_function_5,
            "easom_function": self.easom_function,
            "michalewicz_function": self.michalewicz_function,
            "beale_function": self.beale_function,
            "himmelblau": self.himmelblau,
            "rosenbrock": self.rosenbrock
        }

        if problem_name in self.problems:
            self.problem = self.problems[problem_name]
        else:
            raise ValueError(f"Problem '{problem_name}' is not defined.")

    def evaluate_fitness(self, i):
        return self.problem(i)
    def zdt1(self, population):
        f1 = population[:, 0]
        g = 1 + (9 / (self.num_variables - 1)) * np.sum(population[:, 1:], axis=1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.column_stack((f1, f2))

    def zdt2(self, population):
        f1 = population[:, 0]
        g = 1 + (9 / (self.num_variables - 1)) * np.sum(population[:, 1:], axis=1)
        f2 = g * (1 - np.power(f1 / g, 2))
        return np.column_stack((f1, f2))

    def get_bounds(self):
        """
        Get the bounds for the decision variables.

        :return: Tuple of arrays (lower_bound, upper_bound)
        """
        lower_bound = np.zeros(self.num_variables)
        upper_bound = np.ones(self.num_variables)
        return lower_bound, upper_bound

    def ackley(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi

        if x.ndim == 1:
            # Single individual case
            d = x.shape[0]  # number of dimensions (columns)
            term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
            term2 = -np.exp(np.sum(np.cos(c * x)) / d)
        else:
            # Population case (each row is an individual)
            d = x.shape[1]  # number of dimensions (columns)
            term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / d))
            term2 = -np.exp(np.sum(np.cos(c * x), axis=1) / d)

        return term1 + term2 + a + np.e

    def rosenbrock(self, x):
        # Calculate the fitness (Rosenbrock function)
        x1, x2 = x[0][0], x[0][1]
        fitness = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

        # Initialize penalty
        total_penalty = 0

        # Check if constraints are part of the problem (i.e., problem has constraints)
        # Constraint 1: x1 + x2 <= 2
        g1 = x1 + x2 - 2
        if g1 > 0:  # If constraint is violated
            total_penalty += g1 ** 2  # Apply quadratic penalty

        # Constraint 2: x1 * x2 >= -1
        g2 = -(x1 * x2 + 1)
        if g2 > 0:  # If constraint is violated
            total_penalty += g2 ** 2  # Apply quadratic penalty

        # Return the fitness and the total penalty
        return fitness + total_penalty  # Minimize the objective with penalties for violations

    def himmelblau(self, x):
        x1, x2 = x[0][0], x[0][1]
        return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2

    def bukin(self, x):
        term1 = 100 * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0] ** 2))
        term2 = 0.01 * np.abs(x[:, 0] + 10)
        return term1 + term2

    def cross_in_tray(self, x):
        # Check if x is a 2D array (population) or a 1D array (single individual)
        if x.ndim == 2:
            # x is a 2D array; process the entire population
            results = []
            for individual in x:
                x0, x1 = individual[0], individual[1]
                term1 = np.sin(x0) * np.sin(x1)
                term2 = np.exp(np.abs(100 - np.sqrt(x0 ** 2 + x1 ** 2) / np.pi))
                result = -0.0001 * (np.abs(term1 * term2) + 1) ** 0.1
                results.append(result)
            return np.array(results)
        else:
            # x is a 1D array; process a single individual
            x0, x1 = x[0], x[1]
            term1 = np.sin(x0) * np.sin(x1)
            term2 = np.exp(np.abs(100 - np.sqrt(x0 ** 2 + x1 ** 2) / np.pi))
            result = -0.0001 * (np.abs(term1 * term2) + 1) ** 0.1
            return result

    def drop_wave(x):
        x1, x2 = x
        numerator = - (1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2)))
        denominator = 0.5 * (x1 ** 2 + x2 ** 2) + 2
        return numerator / denominator

    def eggholder(x):
        x1, x2 = x
        term1 = x2 + 47
        term2 = x1 / 2 + (x2 + 47)
        term3 = x1 - (x2 + 47)
        return - (term1 * np.sin(np.sqrt(np.abs(term2)))) - (x1 * np.sin(np.sqrt(np.abs(term3))))

    def gramacy_lee(x):
        term1 = np.sin(10 * np.pi * x) / (2 * x)
        term2 = (x - 1) ** 4
        return term1 + term2

    def griewank(x):
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1

    def holder_table(x):
        term1 = np.sin(x[0]) * np.cos(x[1])
        term2 = np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        return -np.abs(term1 * term2)

    def langermann(x, m=5, c=[1, 2, 5, 2, 3], A=np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])):
        result = 0
        for i in range(m):
            sum_sq = np.sum((x - A[i]) ** 2)
            result += c[i] * np.exp(-sum_sq / np.pi) * np.cos(np.pi * sum_sq)
        return result

    def levy(x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3

    def levy_n13(x):
        term1 = np.sin(3 * np.pi * x[0]) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
        return term1 + term2 + term3

    def rastrigin(x):
        d = len(x)
        return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    def schaffer_n2(x):
        x1, x2 = x
        return 0.5 + (np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

    def schaffer_n4(x):
        x1, x2 = x
        return 0.5 + (np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

    def schwefel(x):
        d = len(x)
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def shubert(x):
        x1, x2 = x
        sum1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in range(1, 6)])
        sum2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in range(1, 6)])
        return sum1 * sum2

    def bohachevsky(x):
        x1, x2 = x
        return x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7

    def perm_function(x, beta=0.5):
        d = len(x)
        outer_sum = 0
        for i in range(d):
            inner_sum = 0
            for j in range(d):
                inner_sum += (j + 1 + beta) * (x[j] ** (i + 1) - 1 / (j + 1) ** (i + 1))
            outer_sum += inner_sum ** 2
        return outer_sum

    def rotated_hyper_ellipsoid(x):
        return np.sum([(np.sum(x[:i + 1] ** 2)) for i in range(len(x))])

    def sphere_function(x):
        return np.sum(x ** 2)

    def sum_of_different_powers(x, p=2):
        return np.sum(np.abs(x) ** p)

    def sum_of_squares(x):
        return np.sum(x ** 2)

    def trid_function(x):
        n = len(x)
        return np.sum((x[:-1] ** 2) * (1 + np.sin(3 * np.pi * x[1:]) ** 2)) + (x[-1] ** 2) * (
                1 + np.sin(3 * np.pi * x[0]) ** 2)

    def booth_function(x):
        x1, x2 = x
        return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

    def matyas_function(x):
        x1, x2 = x
        return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2

    def mccormick_function(x):
        x1, x2 = x
        term1 = np.sin(x1 + x2)
        term2 = np.sin(x1 - x2)
        return np.sin(x1 + x2) + np.sin(x1 - x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1

    def power_sum_function(x, p=2):
        return np.sum(np.abs(x) ** p)

    def zakharov_function(x):
        n = len(x)
        term1 = np.sum(x ** 2)
        term2 = np.sum(0.5 * np.arange(1, n + 1) * x) ** 2
        term3 = np.sum(0.5 * np.arange(1, n + 1) * x) ** 4
        return term1 + term2 + term3

    def three_hump_camel_function(x):
        x1, x2 = x
        return 2 * x1 ** 2 - 1.05 * x1 ** 4 + (x1 ** 2 * x2) + x2 ** 2

    def six_hump_camel_function(x):
        x1, x2 = x
        term1 = (4 - 2.1 * x1 ** 2 + (1 / 3) * x1 ** 4) * x1 ** 2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
        return term1 + term2 + term3

    def dixon_price_function(x):
        n = len(x)
        term1 = (x[0] - 1) ** 2
        term2 = np.sum([(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, n)])
        return term1 + term2

    def rosenbrock_function(x):
        x1, x2 = x
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    def de_jong_function_5(x):
        x1, x2 = x
        term1 = (x1 ** 2 + x2 ** 2) ** 0.5
        return np.sin(term1) ** 2 - 0.5 / (1 + 0.001 * (x1 ** 2 + x2 ** 2))

    def easom_function(x):
        x1, x2 = x
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))

    def michalewicz_function(x, m=10):
        x = np.asarray(x)
        return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x) + 1) * x ** 2) / np.pi) ** (2 * m))

    def beale_function(x):
        x1, x2 = x
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return term1 + term2 + term3

#
