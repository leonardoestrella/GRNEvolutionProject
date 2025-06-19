"""
Support functions for the main script.
"""

import numpy as np
from scipy.stats import norm
from params import config

# Ensemble values

n_genes = config['n_genes'] # probability of a gene being expressed
p_expression = config['p_expression']
c_connection = config['c_connection'] # fraction of connections
p_mutation = 1/ (c_connection * n_genes**2) # probability of mutations

mu = config['mu']
sigma = config['sigma']
rho_w = config['rho_w'](loc = mu, scale = sigma) # weight distribution

# Dynamics and selection

n_max_steps = config['n_max_steps']# maximum number of steps for finding a stable state
selection_strength = config['selection_strength'] # strength of selection in the fitness function
max_attempts = config['max_attempts'] # maximum number of attempts to generate an offspring from two parents

# Class definition

class AdjacencyMatrix:
    def __init__(self, label, n_nodes = 0, weighted_matrix = np.zeros(0),
                 fitness = 0, stable_state = None, target_state = None,
                 initial_state = np.zeros(0), path_length = None):
        
        # Characteristics attributes
        self.label = label # indicator label for the adjacency matrix
        self.n_nodes = n_nodes
        self.weighted_matrix = weighted_matrix

        # Dynamics attributes
        self.initial_state = initial_state #this should be the same for all matrices
        self.stable_state = stable_state

        # Evolution attributes
        self.fitness = fitness # fitness of the matrix
        self.target_state = target_state
        self.path_length = path_length

    # Operations

    def transfer_values(self, other_matrix): 
        """
        Transfer the values from another adjacency matrix to this one conserving
        its own label
        """
        self.label = self.label
        self.n_nodes = other_matrix.n_nodes
        self.weighted_matrix = other_matrix.weighted_matrix
        self.initial_state = other_matrix.initial_state
        self.stable_state = other_matrix.stable_state
        self.fitness = other_matrix.fitness
        self.target_state = other_matrix.target_state
        self.path_length = other_matrix.path_length

    def reset(self, label, n_nodes = 0, weighted_matrix = np.zeros(0),
                 fitness = 0, stable_state = None, target_state = None,
                 initial_state = np.zeros(0), path_length = None):
        """
        Reset the adjacency matrix with new parameters
        """
        # Characteristics attributes
        self.label = label # indicator label for the adjacency matrix
        self.n_nodes = n_nodes 
        self.weighted_matrix = weighted_matrix

        # Dynamics attributes
        self.initial_state = initial_state #this should be the same for all matrices
        self.stable_state = stable_state

        # Evolution attributes
        self.fitness = fitness # fitness of the matrix
        self.target_state = target_state
        self.path_length = path_length


    # Dynamics
    def step_forward(self, current_state, activation_function = np.sign):
        """
        Perform a single step forward in the dynamics of the system using matrix operations.
        Args:
            current_state (np.ndarray): The current state of the system.
            activation_function (function): Activation function to apply to the state.
        Returns:
            np.ndarray: The new state after applying the dynamics.
        """
        
        updated_state = self.weighted_matrix @ current_state
        return activation_function(updated_state)

    def find_stable_state(self,n_steps, activation_function = np.sign):
        """
        Run the dynamics for n_steps until finding a stable state. It stores the
        stable state in the `stable_state` attribute of the object.
        Args:
            n_steps (int): Maximum number of steps to evolve the system.
            activation_function (function): Activation function to apply to the state.
        """
        current_state = np.copy(self.initial_state)
        self.path_length = 0
        for step in range(n_steps):
            new_state = self.step_forward(current_state, activation_function)

            # Check if the new state is stable
            if np.array_equal(new_state, current_state):
                self.stable_state = new_state
                self.path_length = step
                return
            current_state = new_state
        # If we reach here, it means we did not find a stable state within n_steps
        self.stable_state = None
        self.path_length = None

    # Evolution
   
    def compute_fitness(self, distance_function, fitness_function, unstable_fitness):
        """
        Compute the fitness of the adjacency matrix based on the stable state.
        The fitness is defined as the number of nodes in the stable state that match
        the target state.
        Args:
            distance_function (function): Function to compute the distance between states.
            fitness_function (function): Function to compute the fitness based on the distance.
            unstable_fitness (float): Fitness value for unstable states.
        """
        if self.stable_state is None: # Modification required to handle non-constant unstable fitness
            self.fitness = unstable_fitness 
        
        else:
            distance = distance_function(self.stable_state, self.target_state)
            self.fitness = fitness_function(distance)