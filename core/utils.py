from params import config
from scipy.stats import norm
import numpy as np
import csv
import os


n_genes = config['n_genes'] # probability of a gene being expressed
c_connection = config['c_connection'] # fraction of connections
p_mutation = 1/ (c_connection * n_genes**2) # probability of mutations
p_recombination = config['p_recombination']

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
    return np.exp(-distance**2 / selection_strength)

# Offspring generation and selection

def generate_offspring(population_list, old_population,
                       p_recombination=p_recombination, p_mutation=p_mutation,
                       max_attempts=max_attempts, rng=np.random.default_rng(),
                       distance_function=default_distance,
                       fitness_function=default_fitness,
                       unstable_fitness=0, verbose=False):
    """
    Generate the offspring of an old generation by recombining paris of randomly chosen matrices and
    stores them it the current population (if survived).
    * Changes the elements of population_list!
    Args:
        population_list (list): Contains AdjacencyMatrix objects. It will be filled with the offspring
        old_population (list): Contains AdjacencyMatrix objects. It will be used as the parents to generate offspring
        p_recombination (float): Probability of recombination.
        p_mutation (float): Probability of mutation for each edge. 
        max_attempts (int): Number of times it will try to find a child that survives from a pair of parents
        rng (np.random.Generator): Optimized random number generator
        distance_function (function): Computes a distance measure between two vectors
        fitness_function (function): Computes a fitness value (from [0,1]) from a distance measure
        unstable_fitness(float): Minimum fitness assigned to an unstable child
        verbose (bool): If true, prints some stages
    """
    if population_list is None or old_population is None:
        raise ValueError("Both population_list and old_population are required.")

    n_children = len(population_list)
    n_nodes    = old_population[0].n_nodes
    N_parents  = len(old_population)

    # --- draw all parents at once -------------------------------------------------
    pa_idx = rng.integers(0, N_parents, size=n_children)
    pb_idx = rng.integers(0, N_parents, size=n_children)

    for k, child in enumerate(population_list):
        attempts = 0
        survived = False
        pa = old_population[pa_idx[k]]
        pb = old_population[pb_idx[k]]

        while not survived and attempts < max_attempts:

            A = pa.weighted_matrix
            B = pb.weighted_matrix

            # -------- recombination ----------------------------------
            new_adj = A.copy()
            if p_recombination:
                row_mask = rng.random(n_nodes) < p_recombination # False values
                # indicate a recombination happened
                new_adj[row_mask] = B[row_mask]

            # -------- mutation ---------------------------------------
            if p_mutation:
                mut_mask = (new_adj != 0) & (rng.random(new_adj.shape) < p_mutation)
                if mut_mask.any():
                    new_adj[mut_mask] = rho_w.rvs(mut_mask.sum())

            # -------- update & fitness --------------------------------------------
            child.reset(label=child.label, n_nodes=n_nodes,
                        weighted_matrix=new_adj, fitness=0,
                        stable_state=None, target_state=pa.target_state,
                        initial_state=pa.initial_state)

            child.find_stable_state(n_steps=n_max_steps) # This is the bottleneck!
            child.compute_fitness(distance_function, fitness_function, unstable_fitness)

            survived = rng.random() < child.fitness
            attempts += 1

            if not survived and attempts >= max_attempts:
                if verbose:
                    print(f"Child {k}: max attempts reached from {pa.label} & {pb.label}. Redrawing.")
                pa = old_population[rng.integers(0, N_parents)].weighted_matrix
                pb = old_population[rng.integers(0, N_parents)].weighted_matrix
                attempts = 0

def clear_csv_file_with_headers(csv_filename,headers):
    """
    Clears a csv and adds headers
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

def store_data_instant(objects,csv_filename,generation):
    """
    Generates a csv data collected from a list of objects at a
    particular instance.
    Args:
        objects (list of AdjacencyMatrix): objects to take statistics from
        csv_filename (str): name of the output file
        generation (int): timestep associated with this recording 
    """
    n_nets = len(objects)
    n_genes = objects[0].weighted_matrix.shape[0]
    fitness_values = [mat.fitness for mat in objects]
    # matrices = [mat.weighted_matrix for mat in objects]

    unstable_states = sum(1 for mat in objects if mat.stable_state is None)

    stats = {
        'generation': generation,
        'fitness_mean': np.mean(fitness_values),
        'fitness_min': np.min(fitness_values),
        'fitness_max': np.max(fitness_values),
        'fitness_std': np.std(fitness_values),

        'fitness_p5': np.percentile(fitness_values,5),
        'fitness_p95': np.percentile(fitness_values,95),
        'fitness_p10': np.percentile(fitness_values,10),
        'fitness_p25': np.percentile(fitness_values,25),
        'fitness_p75': np.percentile(fitness_values,75),
        'fitness_p90': np.percentile(fitness_values,90),

        'unstable_states': unstable_states,
        'n_nets': n_nets,
        'n_genes': n_genes
    }

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())

        # Write header only if file did not exist before
        if not file_exists:
            writer.writeheader()

        writer.writerow(stats)

    # matrix operations and perturbations

# def store_data_10gen(objects, csv_filename, timestep)

    

## Future functions

# Perturbations functions

# def mutation_perturbation():

# def orthogonal_perturbation():


# Analysis functions

# def path_length():

