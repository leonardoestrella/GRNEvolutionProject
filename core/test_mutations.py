from utils import mutation_perturbation
import numpy as np
from scipy.stats import norm
from adjacency_class import AdjacencyMatrix


np.random.seed(42)

tested_matrix = np.array([[1,0,0], 
                          [0,1,0], 
                          [0,0,1]], dtype = float)

rho_w = norm(loc = 0, scale = 1)
print(rho_w.rvs())

print(mutation_perturbation(tested_matrix, rho_w))

random_matrix = rho_w.rvs(size = (3,3))
print(random_matrix)
print(mutation_perturbation(random_matrix, rho_w))

# Perturbations work as intended!

# Testing stability stuff

# Finding a stable matrix
A = AdjacencyMatrix(label = 'Testing', weighted_matrix=random_matrix, n_nodes = 3,
                    initial_state = np.array([1,-1,-1]))
A.find_stable_state(n_steps = 1000)
count = 0
while A.stable_state is None and count < 1000:
    random_matrix = rho_w.rvs(size=(3,3))
    A.reset(label = A.label, weighted_matrix = random_matrix, initial_state=A.initial_state)
    A.find_stable_state(n_steps = 1000)
    count += 1 

print(A.stable_state)

# Making 10 perturbations
B = AdjacencyMatrix(label = 'Perturbed', n_nodes = 3, initial_state=A.initial_state)
count_stable = 0
for n_pert in range(10):
    perturbed_matrix = mutation_perturbation(A.weighted_matrix, rho_w)
    B.reset(label = B.label, weighted_matrix= perturbed_matrix, initial_state=B.initial_state)
    B.find_stable_state(n_steps = 1000)
    if B.stable_state is not None and np.array_equal(B.stable_state, A.stable_state):
        count_stable += 1

print(f"Number of stable perturbations: {count_stable} out of 10") 
