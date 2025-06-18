"""
Carry out the evolutionary algorithm using the functions in
adjacency_class.py and utils.py, and the parameters in params.py.
It records the data about the adjacency matrix in a separate file.

NOTE (June 16): The code seems to be inefficient. We would like to 
make it run faster. The main driver of running time is the 
generate_offspring function. We also want to add more statistics and
test other parameter values. Maybe I Can move the parameters into this file? 
Finally, I need to create the output file. I am sure the way in which 
I am managing the different files and functions is not the best...
"""

import numpy as np
from scipy.stats import norm
from params import config
from adjacency_class import AdjacencyMatrix
from utils import default_distance, default_fitness, generate_offspring, store_data_instant, clear_csv_file_with_headers
import matplotlib.pyplot as plt

# Write down the parameters

## Ensemble values

n_genes = config['n_genes'] # probability of a gene being expressed
p_expression = config['p_expression']
c_connection = config['c_connection'] # fraction of connections
p_mutation = 1/ (c_connection * n_genes**2) # probability of mutations

mu = config['mu']
sigma = config['sigma']
rho_w = config['rho_w'](loc = mu, scale = sigma) # weight distribution

## Dynamics and selection

n_max_steps = config['n_max_steps']# maximum number of steps for finding a stable state
selection_strength = config['selection_strength'] # strength of selection in the fitness function
max_attempts = config['max_attempts'] # maximum number of attempts to generate an offspring from two parents
p_recombination = config['p_recombination']

## Size of the system and simulation steps

N_nets = config['N_nets']
n_generations = config['n_generations']

# Initialize the system

target_state = np.random.choice([1,-1], size = (n_genes))
initial_state = np.random.choice([1,-1], size = (n_genes))


headers = [
        'generation',
        'fitness_mean', 'fitness_min', 'fitness_max', 'fitness_std',
        'fitness_p5', 'fitness_p10', 'fitness_p25', 'fitness_p75', 'fitness_p90', 'fitness_p95',
        'unstable_states', 'n_nets', 'n_genes'
    ]
clear_csv_file_with_headers("analysis/summary_stats.csv", headers) # Clear csv

current_pop = []
old_pop = []

for i in range(N_nets):
    # Initialize random matrices from samples of a randomly sampled matrix
    # Deviation from Wagner! -- I am not imposing a founder population. Let's
    # see if that allows for convergence
    random_matrix = rho_w.rvs(size = (n_genes,n_genes))
    old_pop.append(AdjacencyMatrix(label = f'O{i}', n_nodes = n_genes,
                                       weighted_matrix= random_matrix, fitness = None,
                                       stable_state = None, target_state = target_state,
                                       initial_state = initial_state))
    current_pop.append(AdjacencyMatrix(label = f"C{i}"))

# Evolution: Loop in stages

## Data storage - should be split into a different file and
## data-generation function given the list of nodes. 
avg_fitness_hist = np.zeros(n_generations+1)
std_fitness_hist = np.zeros(n_generations+1)
unstable_states = np.zeros(n_generations+1)

## Calculate initial statistics

for mat in old_pop:
    mat.find_stable_state(n_max_steps,activation_function = np.sign)
    mat.compute_fitness(default_distance,default_fitness, np.exp(-1/selection_strength))

store_data_instant(old_pop,'analysis/summary_stats.csv',0)

for generation in range(n_generations):

    # Generate offspring from the generation
    # Fitness calculation is already in the generate offspring function
    generate_offspring(population_list = current_pop, old_population = old_pop,
                       p_recombination=p_recombination, max_attempts = max_attempts,
                       p_mutation=p_mutation, distance_function=default_distance,
                       fitness_function= default_fitness, unstable_fitness= np.exp(-1/selection_strength))
    # Bottleneck in this function. find_stable_state takes a lot of time. 
    # Might need to change the whole function to introduce parallelization!

    # Store instantaneous data

    store_data_instant(current_pop, 'analysis/summary_stats.csv', generation+1)

    ## Number of unstable states
    unstable_states[generation+1] = sum(1 for mat in current_pop if mat.stable_state is None) # O(N)

    # Transfer offspring into old list and clear list
    for idx in range(N_nets): # Complexity O(N)
        old_pop[idx].transfer_values(current_pop[idx])
        current_pop[idx].reset(label = current_pop[idx].label)