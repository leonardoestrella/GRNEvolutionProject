import numpy as np
from adjacency_class import AdjacencyMatrix

"""
Tests for the AdjacencyMatrix Class
Testing:
1. Creation of objects
2. Taking a step forward
3. Find its stable state (if any)
4. Computing fitness
"""

# 1 Creation of objects

a_1 = np.array([[1,0],
                [0,1]])
a_2 = np.array([[0,1],
                [1,0]])
a_3 = np.array([[1,0,0],
                [-1,0,1],
                [0,-1,1]])

matrix_a1 = AdjacencyMatrix(label="A1", n_nodes=2, weighted_matrix=a_1,
                          fitness=None, stable_state=None,
                          target_state=np.array([1, -1]),
                          initial_state=np.array([1, -1]))

matrix_a2 = AdjacencyMatrix(label="A2", n_nodes=2, weighted_matrix=a_2,
                          fitness=None, stable_state=None,
                          target_state=np.array([1, -1]),
                          initial_state=np.array([-1, 1]))

matrix_a3 = AdjacencyMatrix(label="A3", n_nodes=3, weighted_matrix=a_3,
                          fitness=None, stable_state=None,
                          target_state=np.array([1, -1, 1]),
                          initial_state=np.array([1, -1, -1]))

print(f"""Three AdjacencyMatrix objects created:\n 
      {matrix_a1.label} = \n{matrix_a1.weigthed_matrix}\n
      {matrix_a2.label} = \n{matrix_a2.weigthed_matrix}\n
      {matrix_a3.label} = \n{matrix_a3.weigthed_matrix}\n""")

# 2. Taking a step forward

# Initial states
s1 = np.array([1,-1])
s2 = np.array([-1,1])
s3 = np.array([1/2,1/2])

print("Making tests in dynamics")

assert( np.array_equal(matrix_a1.step_forward(s1, activation_function=np.sign), s1))
assert( np.array_equal(matrix_a1.step_forward(s2, activation_function=np.sign), s2))
assert( np.array_equal(matrix_a1.step_forward(s3, activation_function=np.sign), np.array([1,1])))
assert( np.array_equal(matrix_a2.step_forward(s1, activation_function=np.sign), np.array([-1,1])))
assert( np.array_equal(matrix_a2.step_forward(s2, activation_function=np.sign), np.array([1,-1])))
assert( np.array_equal(matrix_a2.step_forward(s3, activation_function=np.sign), np.array([1,1])))

s4 = np.array([1,-1,1])
assert( np.array_equal(matrix_a3.step_forward(s4,activation_function=np.sign), np.sign(a_3 @ s4)))

print("All tests passed!")

# 3. Finding stable states
print("\nTesting stable states and fitnesses")

matrix_a1.find_stable_state(n_steps = 100, activation_function = np.sign)
matrix_a2.find_stable_state(n_steps = 100, activation_function = np.sign)
matrix_a3.find_stable_state(n_steps = 100, activation_function = np.sign)

assert(np.array_equal(matrix_a1.stable_state, matrix_a1.initial_state))
assert(matrix_a2.stable_state == None)
assert(matrix_a3.stable_state is not None) # For curiosity, stable state is [1,0,1] given the initial state

# 4. Computing fitness

from utils import default_distance, default_fitness
from params import config
selection_strength = config['selection_strength']

matrix_a1.compute_fitness(distance_function=default_distance, fitness_function=default_fitness,unstable_fitness = np.exp(-1/selection_strength))
matrix_a2.compute_fitness(distance_function=default_distance, fitness_function=default_fitness,unstable_fitness = np.exp(-1/selection_strength))
matrix_a3.compute_fitness(distance_function=default_distance, fitness_function=default_fitness,unstable_fitness = np.exp(-1/selection_strength))

assert( matrix_a1.fitness is not None)
assert( matrix_a2.fitness is not None)
assert( matrix_a3.fitness is not None)

print("All tests passed!")