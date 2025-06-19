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
from utils import clear_csv_file_with_headers, run_simulation, store_data_comparison, store_data_distributions


# how many times the simulation will run
n_simulations = config['n_simulations']
n_genes = config['n_genes']

headers_comparison = ['mean_path_initial', 'std_path_initial', 'se_path_initial', 'mean_path_final',
                      'std_path_final','se_path_final', 'perc_completion_initial','perc_completion_final',
                      'mutation_stability_initial', 'mutation_stability_final']
headers_timeseries = ['generation', 'fitness_mean', ' fitness_std', 'fitness_std', 'fitness_se', 'unstable_states',
                      'perc_unstable', 'path_mean', 'path_std', 'path_se']
headers_distributions = []

clear_csv_file_with_headers("analysis/initial_final.csv", headers_comparison) # Clear csv
clear_csv_file_with_headers("analysis/timeseries.csv", headers_timeseries)
clear_csv_file_with_headers("analysis/distributions.csv", headers_distributions)

timeseries = "analysis/timeseries.csv"
distributions = "analysis/distributions.csv"

repetitions = 10 # To be put in the parameters file

for simulation in range(n_simulations):

    # Initialize the system

    target_state = np.random.choice([1,-1], size = (n_genes))
    initial_state = np.random.choice([1,-1], size = (n_genes))

    initial_pop, final_pop = run_simulation(config, timeseries_filename=timeseries)
    timeseries = None

    rho_w = config['rho_w'](loc = config['mu'], scale = config['sigma'])

    store_data_comparison(initial_pop, final_pop, "analysis/initial_final.csv", rho_w, repetitions)
        ## path_lenghts
        ## mutational and orthogonal stability 
    if distributions is not None:
        store_data_distributions(initial_pop, final_pop, distributions)
        distributions = None
