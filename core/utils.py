from params import config
from scipy.stats import norm
import numpy as np
import csv
import os
from adjacency_class import AdjacencyMatrix
import pandas as pd

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

def default_fitness(distance, selection_strength = 0.1):
    """
    Compute the fitness based on the distance.
    """
    return np.exp(-distance**2 / selection_strength)

# Perturbations functions

def mutation_perturbation(matrix, rho_w):
    """
    Creates a matrix with a mutation perturbation with a non-zero entry
    resampled from the distribution rho_w
    Args:
        matrix (np.array): weighted adjacency matrix
        rho_w (scipy probability distribution): distribution from where weights are taken
    Returns:
        np.array: perturbed weighted adjacency matrix
    """
    non_zero_indices = np.argwhere(matrix != 0) # can be optimized
    perturbed_mat = matrix.copy()

    if non_zero_indices.size == 0:
        return np.zeros_like(matrix)

    # Randomly choose one of the non-zero indices
    #print(f"Unperturbed matrix:\n{perturbed_mat}")
    i, j = non_zero_indices[np.random.choice(len(non_zero_indices))]
    resample = rho_w.rvs()
    #print(f"Chosen indexes{i,j}")
    #print(f"Resampled weight: {resample}")

    perturbed_mat[i,j] = resample
    #print(f"Perturbed matrix: {perturbed_mat}")
    
    return perturbed_mat

def pop_mutation_perturbation(pop,rho_w, repetitions, n_max_steps):
    """
    Computes the proportion of mutation perturbations that lead the stable
    state unchanged
    * NOTE June 19 - This is highly under-optimized! ChatGPT does a bad job
    at trying to do so because does not understand the context.
    Args:
        pop (list of AdjacencyMatrix): population to compute the perturbations
        rho_w (simpy probability distribution): weight distribution
        repetitions (int): number of perturbations per matrix
        n_max_steps (int): maximum number of steps to find a stable state
    Returns:
        float: proportion of perturbations that left the stable states unchanged
    """
    count = 0
    for rep in range(repetitions):
        dummy_matrix = AdjacencyMatrix(label = f'dummy{rep}')
        for mat in pop:
            if mat.stable_state is None: 
                # Skip this matrix. If the matrix is unstable, then a mutation producing an stable matrix
                # does not yield an epigenetic matrix. 
                continue 
            dummy_matrix.reset(label = dummy_matrix.label, n_nodes = mat.n_nodes,
                               weighted_matrix = np.zeros((mat.n_nodes, mat.n_nodes)),
                               fitness = 0, stable_state = None,
                               target_state = mat.target_state,
                               initial_state = mat.initial_state,
                               path_length = 0)
            dummy_matrix.weighted_matrix = mutation_perturbation(mat.weighted_matrix, rho_w)
            dummy_matrix.find_stable_state(n_steps = n_max_steps)
            if np.array_equal(dummy_matrix.stable_state, mat.stable_state):
                count += 1
    return count / (repetitions * len(pop))

# def orthogonal_perturbation():

# Data stuff

def clear_csv_file_with_headers(csv_filename,headers):
    """
    Clears a csv and adds headers
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

def store_data_timeseries(objects,csv_filename,generation):
    """
    Adds rows to a csv file from a list of objects per generation
    Args:
        objects (list of AdjacencyMatrix): objects to take statistics from
        csv_filename (str): name of the output file
        generation (int): timestep associated with this recording 
    """
    n_nets = len(objects)
    #n_genes = objects[0].weighted_matrix.shape[0]
    fitness_values = [mat.fitness for mat in objects]
    path_lengths = [mat.path_length for mat in objects if mat.stable_state is not None]

    # matrices = [mat.weighted_matrix for mat in objects]

    unstable_states = sum(1 for mat in objects if mat.stable_state is None)

    stats = {
        'generation': generation,
        'fitness_mean': np.mean(fitness_values),
        'fitness_std': np.std(fitness_values),
        'fitness_se': np.std(fitness_values) / np.sqrt(n_nets),

        'unstable_states': unstable_states,
        'perc_unstable': unstable_states / n_nets,

        'path_mean': np.mean(path_lengths),
        'path_std': np.std(path_lengths),
        'path_se': np.std(path_lengths) / np.sqrt(n_nets)
    }

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        # Write header only if file did not exist before
        if not file_exists:
            writer.writeheader()

        writer.writerow(stats)

def store_data_comparison(initial,final, csv_filename, rho_w, repetitions, n_max_steps):
    n_genes = initial[0].n_nodes
    # mutational stability

    initital_mutation_stability = pop_mutation_perturbation(initial,rho_w, repetitions, n_max_steps)
    final_mutation_stability = pop_mutation_perturbation(final, rho_w, repetitions, n_max_steps)

    # orthogonal stability - June 18

    # path lengths
    initial_path = [mat.path_length for mat in initial]
    initial_complete_path = [mat.path_length for mat in initial if mat.path_length is not None]
    final_path = [mat.path_length for mat in final]
    final_complete_path = [mat.path_length for mat in final if mat.path_length is not None]

    stats = {
        'mean_path_initial': np.mean(initial_complete_path),
        'std_path_initial': np.std(initial_complete_path),
        'se_path_initial': np.std(initial_complete_path) / np.sqrt(n_genes),

        'mean_path_final': np.mean(final_complete_path),
        'std_path_final': np.std(final_complete_path),
        'se_path_final': np.std(final_complete_path) / np.sqrt(n_genes),

        'perc_completion_initial': sum(1 for path in initial_path if path is not None) / len(initial_path),
        'perc_completion_final': sum(1 for path in final_path if path is not None) / len(final_path),


        'mutation_stability_initial': initital_mutation_stability,
        'mutation_stability_final': final_mutation_stability    
    }

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        # Write header only if file did not exist before
        if not file_exists:
            writer.writeheader()

        writer.writerow(stats)
    
def store_data_distributions(initial,final, csv_filename):
    # path lengths
    path_initial = np.array([mat.path_length for mat in initial])
    path_final = np.array([mat.path_length for mat in final])

    df = pd.DataFrame()
    df['path_initial'] = path_initial
    df['path_final'] = path_final

    df.to_csv(csv_filename)    

# Offspring generation and selection

def find_random_stable_matrix(matrix, rho_w, mask,n_max_steps):
    """
    Finds a stable matrices by generating random fully connected matrices
    with weights from a normal distribution N(0,1)
    Args:
        matrix (AdjacencyMatrix): Object to store the stable matrix in
        rho_w (scipy distribution): distribution to sample the edge weights (default Normal(0,1))
        mask (np.array): binary matrix indicating the presence of edges
    """
    n_genes = matrix.n_nodes
    while matrix.stable_state is None:
        random_matrix = rho_w.rvs(size = (n_genes,n_genes)) * mask
        matrix.reset(label = matrix.label, n_nodes = n_genes,
                     weighted_matrix=random_matrix, fitness=0,
                     stable_state=None, target_state=matrix.target_state,
                     initial_state=matrix.initial_state, path_length = 0)
        matrix.find_stable_state(n_steps=n_max_steps)

def initialize_population(initial_pop, old_pop, current_pop, mode, config, initial_state, target_state):
    """
    Args:
        initial_pop, old_pop, current_pop (list): lists to be filled with AdjacencyMatrix objects
        mode (str): 'full' for fully connected, 'sparse' for a given connection
            probability (c_connection in config)
        config (dict): dictionary with the simulation parameters
        initial_state,target_state (np.array): initial and target states for the evolution of GRNs
    """
        # Parameters
    n_genes = config['n_genes'] # probability of a gene being expressed
    N_nets = config['N_nets'] # how many nets are in the population
    c_connection = config['c_connection'] # fraction of connections

    mu = config['mu']
    sigma = config['sigma']
    rho_w = config['rho_w'](loc = mu, scale = sigma) # weight distribution

    n_max_steps = config['n_max_steps'] # maximum number of stpes to find a stable state

    if mode == "full":
        for i in range(N_nets):
            # No founder population!
            normal_matrix = rho_w.rvs(size = (n_genes,n_genes))

            initial_pop.append(AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= normal_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= normal_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
    elif mode == "sparse":
        for i in range(N_nets):
            # No founder population!

            adjacency_matrix = np.random.choice([0, 1], size=(n_genes, n_genes), p=[1 - c_connection, c_connection])
            normal_matrix = rho_w.rvs(size = (n_genes,n_genes))
            random_matrix = adjacency_matrix * normal_matrix

            initial_pop.append(AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= random_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= random_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
    elif mode == "stable,full":
        for i in range(N_nets):
            # Stable initial population
            normal_matrix = rho_w.rvs(size = (n_genes,n_genes))

            matrix = AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= normal_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state)
            find_random_stable_matrix(matrix,rho_w, np.ones((n_genes,n_genes)), n_max_steps)

            initial_pop.append(matrix)
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= matrix.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
    elif mode == "stable,sparse":
        for i in range(N_nets):    # Stable initial population, sparse matrices   
            adjacency_matrix = np.random.choice([0, 1], size=(n_genes, n_genes), p=[1 - c_connection, c_connection])
            normal_matrix = rho_w.rvs(size = (n_genes,n_genes))
            random_matrix = adjacency_matrix * normal_matrix

            matrix = AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= random_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state)
            find_random_stable_matrix(matrix,rho_w, np.ones((n_genes,n_genes)), n_max_steps)

            initial_pop.append(matrix)
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= matrix.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
        
    elif mode == "founder,full":
        normal_matrix = rho_w.rvs(size = (n_genes,n_genes))
        founder = AdjacencyMatrix(label = 'I0', n_nodes = n_genes,
                                            weighted_matrix= normal_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state)
        find_random_stable_matrix(founder,rho_w, np.ones((n_genes,n_genes)), n_max_steps)
        founder.target_state = founder.stable_state
        for i in range(N_nets):
            initial_pop.append(AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= founder.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = founder.target_state,
                                            initial_state = founder.initial_state))
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= founder.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = founder.target_state,
                                            initial_state = founder.initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
    elif mode == "founder,sparse":

        adjacency_matrix = np.random.choice([0, 1], size=(n_genes, n_genes), p=[1 - c_connection, c_connection])
        normal_matrix = rho_w.rvs(size = (n_genes,n_genes))
        random_matrix = adjacency_matrix * normal_matrix

        founder = AdjacencyMatrix(label = 'founder', n_nodes = n_genes,
                                            weighted_matrix= random_matrix, fitness = None,
                                            stable_state = None, target_state = target_state,
                                            initial_state = initial_state)
        
        find_random_stable_matrix(founder,rho_w, np.ones((n_genes,n_genes)), n_max_steps)

        founder.target_state = founder.stable_state
        for i in range(N_nets):
            initial_pop.append(AdjacencyMatrix(label = f'I{i}', n_nodes = n_genes,
                                            weighted_matrix= founder.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = founder.target_state,
                                            initial_state = founder.initial_state))
            old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                            weighted_matrix= founder.weighted_matrix, fitness = None,
                                            stable_state = None, target_state = founder.target_state,
                                            initial_state = founder.initial_state))
            current_pop.append(AdjacencyMatrix(label = f"C{i}"))
    else: 
        raise ValueError("Initialization mode not recognized")

def generate_offspring(population_list, old_population,
                       p_recombination=0.5, p_mutation=0.1, n_max_steps = 100,
                       max_attempts=100, rng=np.random.default_rng(), rho_w = norm(loc = 0, scale = 1),
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
        n_max_steps(int): Maximum number of steps to find a stable state
        rho_w (scipy distribution): distribution to sample the edge weights (default Normal(0,1))
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
                        initial_state=pa.initial_state, path_length = 0)

            child.find_stable_state(n_steps=n_max_steps) # This is the bottleneck!
            child.compute_fitness(distance_function, fitness_function, unstable_fitness)

            survived = rng.random() < child.fitness
            attempts += 1

            if not survived and attempts >= max_attempts:
                if verbose:
                    print(f"Child {k}: max attempts reached from {pa.label} & {pb.label}. Redrawing.")
                pa = old_population[rng.integers(0, N_parents)]
                pb = old_population[rng.integers(0, N_parents)]
                attempts = 0

def run_simulation(config, initial_state, target_state, timeseries_filename=False): 
    """
    Runs the simulations given a set of parameters and returns the initial
    and final populations. 
        *June 17 - allow for not-fully connected matrix generator
    Args
        config (dict): dictionary with the simulation parameters
        timeseries_filename(str or bool): 
            If a string, it will store the data into a csv file of the timeseries data.
            It won't do anything otherwise
        initial_state, target_state (np.array): initial and target states for the evolution of GRNs
    Returns
        initial_objects, final_objects (lists): lists of objects corresponding to the inital and
            final populations of the GRNs
    """

    # Parameters
    n_genes = config['n_genes'] # probability of a gene being expressed
    N_nets = config['N_nets'] # how many nets are in the population
    n_generations = config['n_generations'] # generations per simulation
    c_connection = config['c_connection'] # fraction of connections
    p_mutation = 1/ (c_connection * n_genes**2) # probability of mutations
    p_recombination = config['p_recombination']

    mu = config['mu']
    sigma = config['sigma']
    rho_w = config['rho_w'](loc = mu, scale = sigma) # weight distribution

    n_max_steps = config['n_max_steps']# maximum number of steps for finding a stable state
    selection_strength = config['selection_strength'] # strength of selection in the fitness function
    max_attempts = config['max_attempts'] # maximum number of attempts to generate an offspring from two parents
    init_mode = config['init_mode']
    initial_pop = [] # Store a copy of the inital configuration!
    current_pop = []
    old_pop = []

    initialize_population(initial_pop, old_pop, current_pop, init_mode, config, initial_state, target_state)

    # Evolution: Loop in stages

    ## Calculate initial statistics

    for mat in old_pop:
        mat.find_stable_state(n_max_steps,activation_function = np.sign)
        mat.compute_fitness(default_distance,default_fitness, np.exp(-1/selection_strength))

    for mat in initial_pop:
        mat.find_stable_state(n_max_steps,activation_function = np.sign)
        mat.compute_fitness(default_distance,default_fitness, np.exp(-1/selection_strength))

    if timeseries_filename is not None:
        store_data_timeseries(old_pop,timeseries_filename,0)

    for generation in range(n_generations):
        if generation % 40 == 0:
            print(f"\t Generation {generation+1} out of {n_generations}")

        # Generate offspring from the generation
        # Fitness calculation is already in the generate offspring function
        generate_offspring(population_list = current_pop, old_population = old_pop,
                        p_recombination=p_recombination, max_attempts = max_attempts,
                        p_mutation=p_mutation, distance_function=default_distance,
                        fitness_function= default_fitness, unstable_fitness= np.exp(-1/selection_strength))
        # Bottleneck in this function. find_stable_state takes a lot of time. 
        # Might need to change the whole function to introduce parallelization!
        if timeseries_filename is not None:
            store_data_timeseries(current_pop, timeseries_filename, generation+1)

        # Transfer offspring into old list and clear list
        for idx in range(N_nets): # Complexity O(N)
            old_pop[idx].transfer_values(current_pop[idx])
            current_pop[idx].reset(label = current_pop[idx].label)

    return initial_pop, old_pop


