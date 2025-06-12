
"""
Parameters
    - N_nodes: Number of nodes in the system
    - n_steps: Maximum number of steps to evolve the system
    - n_matrices: number of adjacency matrices to evolve
    - f: Activation function (default: np.sign)
    - c: Constant for mutation probability (default: 0.5)

## NOTE (June 12) Place new parameters added in the adjacency_class.py in this field too

Initialize system
    - Define optimal state
    - Define initial state
    - Define founder adjacency matrix (and its copies)

Update adjacency matrices -- Developed in June 12
    1. Recombination - (probability of recombination = 0.5)
        From the population, pick a pair of matrices and recombine them. Each
        new combination is subject to the following steps
    2. Mutation - (probability of mutation = 1 / (c * N_nodes**2), where c is a constant)
        Each non-zero element in the adjacency matrix has a probability of mutation. If it mutates,
        we draw from the same distribution as the original ensemble. 
    3. Fitness calculation
    4. Survival test
"""