from params import config
from scipy.stats import norm
import numpy as np

n_genes = config['n_genes'] # probability of a gene being expressed
c_connection = config['c_connection'] # fraction of connections
p_mutation = 1/ (c_connection * n_genes**2) # probability of mutations

mu = config['mu']
sigma = config['sigma']
rho_w = config['rho_w'](loc = mu, scale = sigma) # weight distribution

# Dynamics and selection

n_max_steps = config['n_max_steps']# maximum number of steps for finding a stable state
selection_strength = config['selection_strength'] # strength of selection in the fitness function
max_attempts = config['max_attempts'] # maximum number of attempts to generate an offspring from two parents

# Distances

def default_distance(s1, s2):
    """
    Compute the Hamming distance between two states.
    """
    if s1.shape != s2.shape:
        raise ValueError("s1 and s2 must be the same shape")

    N = s1.size
    return 0.5 - (np.dot(s1, s2) / (2 * N))

# Fitnesses

def default_fitness(distance):
    """
    Compute the fitness based on the distance.
    """
    return np.exp(- distance**2 / selection_strength)

# Offspring generation and selection

def generate_offspring(population_list = None, old_population = None,
                        p_recombination=0.5, max_attempts = max_attempts,
                        p_mutation=p_mutation, distance_function = default_distance,
                        fitness_function = default_fitness, unstable_fitness = 1):
    """
    Generate the offspring of an old generation by recombining paris of randomly chosen matrices and
    stores them it the current population (if survived).
    Args:
        matrix_a, matrix_b (AdjacencyMatrix): Adjacency matrices to recombine.
        p_recombination (float): Probability of recombination. Default = 0.5
        p_mutation (float): Probability of mutation for each edge. Default = 0.1
        population_list (list): List of new population adjacency matrices.
        old_population (list): List of old adjacency matrices in the population.
    Returns:
        AdjacencyMatrix: A new adjacency matrix representing the offspring.

        *** Store evolutionary timeline
    """
    # Requires to do these steps berfore calling the function:
    # 1) Transfer all values from popuLation_list to old_population
    # 2) Clear all values from population_list
    # In this way, the function has as inputs a list of vessels to be filled with children
    # in the current population and a list of parents from the previous generation.

    if population_list is None:
        raise ValueError("The population_list must be provided for offspring generation.")
    if old_population is None:
        raise ValueError("The old_population must be provided for offspring generation.")

    for child in population_list:
        matrix_a = np.random.choice(old_population) # choose two random parents from the old population
        matrix_b = np.random.choice(old_population)
        survived = False
        counter = 0
        while not survived:
            new_adjacency_matrix = matrix_a.weigthed_matrix.copy()

            # Recombination
            if p_recombination > 0:
                # Recombine rows of matrix_a and matrix_b
                for row_idx in range(matrix_a.n_nodes):
                    if np.random.rand() < p_recombination:
                        new_adjacency_matrix[row_idx, :] = matrix_b.weigthed_matrix[row_idx, :]
            # Mutation
            if p_mutation > 0:
                for row_idx, col_idx in np.ndindex(new_adjacency_matrix.shape):
                    if new_adjacency_matrix[row_idx, col_idx] != 0:
                        # Mutate non-zero weights with a certain probability
                        if np.random.rand() < p_mutation:
                            # Mutate the weight by drawing from a normal distribution
                            new_adjacency_matrix[row_idx, col_idx] = rho_w.rvs()

            # Update the current adjacency matrix instance for the offspring
            child.reset(label = child.label, n_nodes = matrix_a.n_nodes,
                        weighted_matrix = new_adjacency_matrix,
                        fitness = 0, stable_state = None,
                        target_state = matrix_a.target_state,
                        initial_state = matrix_a.initial_state)
            
            child.find_stable_state(n_steps=n_max_steps)
            child.compute_fitness(distance_function=distance_function,
                                fitness_function=fitness_function,
                                unstable_fitness=unstable_fitness)
            # Decide survival
            survived = np.random.uniform() < child.fitness
            # It will keep within the while loop until the child survives
            counter += 1
            if not survived and counter >= max_attempts:
                print(f"Max attempts reached from matrices {matrix_a.label} and {matrix_b.label}\n")
                matrix_a = np.random.choice(old_population)
                matrix_b = np.random.choice(old_population)
                print(f"New parents selected: {matrix_a.label} and {matrix_b.label}\n")
                counter = 0


## Future functions

# Perturbations functions

# def mutation_perturbation():

# def orthogonal_perturbation():


# Analysis functions

# def path_length():