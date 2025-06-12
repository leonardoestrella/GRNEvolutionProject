"""
Support functions for the main script.
"""

import numpy as np
from scipy.stats import norm

# Ensemble values

# NOTE (June 12) -- To move parameters into a different file

n_genes = 10
p_expression = 0.2 # probability of a gene being expressed

c = 0.4
p_mutation = 1 / (c * n_genes**2) # probability of a connection (c)

mu, sigma = 0, 1
rho_w = norm(loc = mu, scale = sigma) # distribution of weights

# Parameters

n_max_steps = 1000 # maximum number of steps for finding a stable state
selection_strength = 1 # strength of selection in the fitness function
max_attempts = 100  # maximum number of attempts to generate an offspring from two parents

def default_distance(s1, s2):
    """
    Compute the Hamming distance between two states.
    """
    if s1.shape != s2.shape:
        raise ValueError("s1 and s2 must be the same shape")

    N = s1.size
    return 0.5 - (np.dot(s1, s2) / (2 * N))


def default_fitness(distance):
    """
    Compute the fitness based on the distance.
    """
    return np.exp(- distance**2 / selection_strength)

# Class definition

class AdjacencyMatrix:
    def __init__(self, label, n_nodes = 0, weighted_matrix = np.zeros(0),
                 fitness = 0, stable_state = None, target_state = None,
                 initial_state = np.zeros(0)):
        
        # Characteristics attributes
        self.label = label # indicator label for the adjacency matrix
        self.n_nodes = n_nodes
        self.weigthed_matrix = weighted_matrix

        # Dynamics attributes
        self.initial_state = initial_state #this should be the same for all matrices
        self.stable_state = stable_state

        # Evolution attributes
        self.fitness = fitness # fitness of the matrix
        self.target_state = target_state

    # Operations

    def remove_directed_edge(self, edge): #source is itself
        """
        Remove an edge from the adjacency matrix.
        Args:
            edge (tuple): A tuple representing the edge to be removed (source, target).
        """
        if edge in self.edges:
            self.edges.remove(edge)
        else:
            raise ValueError(f"Edge {edge} not found in the adjacency matrix.")
        
    def add_directed_edge(self, edge):
        """
        Add an edge to the adjacency matrix.
        Args:
            edge (tuple): A tuple representing the edge to be added (source, target).
        """
        if edge not in self.edges:
            self.edges.append(edge)
        else:
            raise ValueError(f"Edge {edge} already exists in the adjacency matrix.")
    
    def transfer_values(self, other_matrix): # PAUSE JUNE 12 - FIGURE OUT HOW WILL THE OFFSPRING GENERATION WORK
        """
        Transfer the values from another adjacency matrix to this one.
        """
        self.label = other_matrix.label
        self.n_nodes = other_matrix.n_nodes
        self.weigthed_matrix = other_matrix.weigthed_matrix
        self.initial_state = other_matrix.initial_state
        self.stable_state = other_matrix.stable_state
        self.fitness = other_matrix.fitness
        self.target_state = other_matrix.target_state

        
    def reset(self, label, n_nodes = 0, weighted_matrix = np.zeros(0),
                 fitness = 0, stable_state = None, target_state = None,
                 initial_state = np.zeros(0)):
        """
        Reset the adjacency matrix with new parameters
        """
        # Characteristics attributes
        self.label = label # indicator label for the adjacency matrix
        self.n_nodes = n_nodes
        self.weigthed_matrix = weighted_matrix

        # Dynamics attributes
        self.initial_state = initial_state #this should be the same for all matrices
        self.stable_state = stable_state

        # Evolution attributes
        self.fitness = fitness # fitness of the matrix
        self.target_state = target_state

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
        
        updated_values = self.weigthed_matrix @ current_state
        return activation_function(updated_values)

    def find_stable_state(self,n_steps, activation_function = np.sign):
        """
        Run the dynamics for n_steps until finding a stable state. It stores the
        stable state in the `stable_state` attribute of the object.
        Args:
            n_steps (int): Maximum number of steps to evolve the system.
            activation_function (function): Activation function to apply to the state.
        """
        current_state = np.copy(self.initial_state)
        for step in range(n_steps):
            new_state = self.step_forward(current_state, activation_function)

            # Check if the new state is stable
            if np.array_equal(new_state, current_state):
                self.stable_state = new_state
                return
            current_state = new_state
        # If we reach here, it means we did not find a stable state within n_steps
        self.stable_state = None

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


## NOTE (June 12) - To turn into assert statements

## Testing some stuff out
## I will likely make a more formal piece of code here...

# Creating a couple of adjacency matrices

#if __name__ == '__main__': # Run the tests only if running precisely this file. 
# Is this professional? 


a_1 = np.array([[1,0],
                [0,1]])
a_2 = np.array([[0,1],
                [1,0]])
a_3 = np.array([[1,0,0],
                [-1,0,1],
                [0,-1,1]])

matrix_a1 = AdjacencyMatrix(label="A1", n_nodes=2, weighted_matrix=a_1,
                          fitness=0, stable_state=None,
                          target_state=np.array([1, -1]),
                          initial_state=np.array([1, -1]))

matrix_a2 = AdjacencyMatrix(label="A2", n_nodes=2, weighted_matrix=a_2,
                          fitness=0, stable_state=None,
                          target_state=np.array([1, -1]),
                          initial_state=np.array([-1, 1]))

matrix_a3 = AdjacencyMatrix(label="A3", n_nodes=3, weighted_matrix=a_3,
                          fitness=0, stable_state=None,
                          target_state=np.array([1, -1, 1]),
                          initial_state=np.array([1, -1, -1]))

# # Making them move a step forward
s1 = np.array([1,-1])
s2 = np.array([-1,1])
s3 = np.array([1/2,1/2])
s_tests = [s1,s2,s3]

print(f"MatrixA1 = \n{matrix_a1.weigthed_matrix}\n")
for s_tested in s_tests:
    print(matrix_a1.step_forward(s_tested, activation_function = np.sign))


print(f"MatrixA2 = \n{matrix_a2.weigthed_matrix}\n")
for s_tested in s_tests:
    print(matrix_a2.step_forward(s_tested, activation_function = np.sign))

s_tests = [np.array([1,0,0]),np.array([1,-1,1]), np.array([0,4,1]) ]
print(f"MatrixA3 = \n{matrix_a3.weigthed_matrix}\n")
for s_tested in s_tests:
    print(f"\nstate tested = {s_tested}\n")
    print(matrix_a3.step_forward(s_tested, activation_function = np.sign))

# ## Everything alright here!

# Finding stable states and Fitness calculations

print("Stable states and Fitness")
matrix_a1.find_stable_state(n_steps = n_max_steps, activation_function = np.sign)
print(matrix_a1.stable_state)
matrix_a1.compute_fitness(distance_function=default_distance, fitness_function=default_fitness,unstable_fitness = np.exp(-1/selection_strength))
print(matrix_a1.fitness)
print(matrix_a1.target_state)

matrix_a2.find_stable_state(n_steps = n_max_steps, activation_function = np.sign)
print(matrix_a2.stable_state) # Shouldn't find a stable state!

matrix_a3.find_stable_state(n_steps = n_max_steps, activation_function = np.sign)
print(matrix_a3.stable_state)

# Testing offspring generation

old_population = [matrix_a1, matrix_a2]
new_population = [AdjacencyMatrix(label = 'dummy1'), AdjacencyMatrix(label = 'dummy2')]
generate_offspring(population_list = new_population, old_population = old_population)

for child in new_population:
    print(child.label, "\n", child.weigthed_matrix, "\n", child.fitness, "\n")

