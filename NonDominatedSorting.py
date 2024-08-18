import numpy as np


class NonDominatedSorting:
    def nds(self, fitness_values):
        s = [[] for i in range(fitness_values.shape[0])]
        n = np.zeros(fitness_values.shape[0])
        rank = []
        fronts = [[]]
        for p in range(fitness_values.shape[0]):
            for q in range(fitness_values.shape[0]):
                if p != q:
                    if np.all(fitness_values[p] <= fitness_values[q]) and np.any(fitness_values[p] < fitness_values[q]):  # if p dominates q then add q to set of solns that p dominates
                        s[p].append(q)
                    else:
                        n[p] = n[p] + 1  # if q dominates then increase the count of solutions that are dominated by p
            # no solution dominates p and p is on the first front
            if n[p] == 0:
                rank[p] = 1
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # Remove the last empty front
        fronts.pop()

        return fronts, rank
