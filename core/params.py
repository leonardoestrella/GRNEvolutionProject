
from scipy.stats import norm

config = {'n_genes': 10, 'p_expression': 0.2, 'c_connection' : 0.4, 
          'mu' : 0, 'sigma' : 1, 'rho_w':  norm, 'n_max_steps': 1000,
          'selection_strength': 1, 'max_attempts': 100}
