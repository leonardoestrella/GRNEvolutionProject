
import numpy as np
from adjacency_class import AdjacencyMatrix
from utils import generate_offspring, default_distance, default_fitness

"""
Tests for functions inside utils.py

Testing:
1. Offspring generation
2. Perturbation functions
"""

# # Testing offspring generation

# Generate initial population and old population as equally-sized lists

a_1 = np.array([[1,0],
                [0,1]])
a_2 = np.array([[0,1],
                [1,0]])
a_3 = np.array([[0.4,0.3],
                [0.6,0.7]])

current_pop = []
old_pop = []

for idx, matrix in enumerate([a_1,a_2,a_3]):
    current_pop.append(AdjacencyMatrix(label = f'Current{idx}'))
    old_pop.append(AdjacencyMatrix(label = f"Old{idx}", n_nodes = 2, weighted_matrix=matrix,
                                fitness = None, stable_state = None, 
                                target_state=np.array([1,-1]),
                                initial_state = np.array([-1,1])))

# Tricked it into never having the same elements via
# ensured mutation

print("Running simple test for offspring generation")

generate_offspring(population_list = current_pop, old_population=old_pop,
                p_recombination=0.5, max_attempts = 100,
                p_mutation = 1.0, distance_function = default_distance,
                fitness_function = default_fitness, unstable_fitness= 1)

for idx, matrix in enumerate(current_pop):
    assert (not np.array_equal(matrix.weighted_matrix, old_pop[idx].weighted_matrix))
    print(matrix.label)
    print(matrix.weighted_matrix)
    # Reset the matrices for other tests
    matrix.reset(label = matrix.label)

print("Simple test passed")