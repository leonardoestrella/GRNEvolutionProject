
from scipy.stats import norm

config = {'n_genes': 5, 'p_expression': 0.2, 'c_connection' : 0.4, 
          'mu' : 0, 'sigma' : 0.1, 'rho_w':  norm, 'n_max_steps': 1000,
          'p_recombination': 0.5,
          'selection_strength': 0.3, 'max_attempts': 100, 'N_nets': 10,
          'n_generations': 100}
