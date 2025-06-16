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
from utils import default_distance, default_fitness, generate_offspring
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

avg_fitness_hist[0] = np.mean([mat.fitness for mat in old_pop])
std_fitness_hist[0] = np.std([mat.fitness for mat in old_pop])
unstable_states[0] = sum(1 for mat in old_pop if mat.stable_state is None)

for generation in range(n_generations):

    # Generate offspring from the generation
    # Fitness calculation is already in the generate offspring function
    generate_offspring(population_list = current_pop, old_population = old_pop,
                       p_recombination=p_recombination, max_attempts = max_attempts,
                       p_mutation=p_mutation, distance_function=default_distance,
                       fitness_function= default_fitness, unstable_fitness= np.exp(-1/selection_strength))
    # Complexity infinitely large? - Randomness make it possible to never end the loop!

    # TO DO: optimize this function!
    # This is the main driver of running time. To find a stable state, we do, at least,
    # one matrix multiplication of complexity O(N^2) for each of the N generated offspring.
    # This contributes to a O(N^3) complexity. 
    
    # Compute data from the new population

    fitness_list = [mat.fitness for mat in current_pop] # O(N)

    avg_fitness = np.mean(fitness_list) #O(N)
    std_fitness = np.std(fitness_list) #O(N)
    avg_fitness_hist[generation+1] = avg_fitness
    std_fitness_hist[generation+1] = std_fitness 

    ## Number of unstable states
    unstable_states[generation+1] = sum(1 for mat in current_pop if mat.stable_state is None) # O(N)

    # Transfer offspring into old list and clear list
    for idx in range(N_nets): # Complexity O(N)
        old_pop[idx].transfer_values(current_pop[idx])
        current_pop[idx].reset(label = current_pop[idx].label)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Average Fitness with Standard Error
axes[0].plot(np.arange(n_generations + 1), avg_fitness_hist, label='Mean', marker='o')
axes[0].fill_between(
    np.arange(n_generations + 1),
    avg_fitness_hist - 2*std_fitness_hist / np.sqrt(N_nets),
    avg_fitness_hist + 2*std_fitness_hist / np.sqrt(N_nets),
    alpha=0.3,
    label='2 SE'
)
axes[0].set_ylabel("Average Fitness")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Number of Unstable States
axes[1].plot(np.arange(n_generations + 1), unstable_states, label='# Unstable States', marker='o', color='orange')
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("# Unstable States")
axes[1].grid(True)

plt.tight_layout()
plt.show()
