module BooleanNetwork

    using Plots, LaTeXStrings, Distributions, DoubleFloats, Plots.PlotMeasures, LinearAlgebra, 
    DataFrames, CSV, LsqFit, StatsBase, Distances, Parameters, Random, StatsBase, Graphs

    export run_simulation
    """
    --------------------
    DYNAMICS
    -------------------
    """

    """
    activation:Function that takes the sign of a number
    """
    function activation(x)
        hard_sign(x) = x ≥ 0 ? 1.0 : -1.0
        return hard_sign.(x)
    end

    """
    distance: Function to compute the distance between two vectors
    with values 1,-1. It clips the vectors to the first N_target elements

    args: 
    - v1,v2: vectors with values 1,-1
    - N_target

    returns:
    - result: a float number between 0 and 1
    """

    function distance(v1, v2, N_target)
        prod = Vector{Float64}(undef,N_target)
        for i in 1:N_target
            prod[i] = v1[i] * v2[i]
        end
        result = 1/2 - 1 / (2N_target) * sum(prod)
        return result
    end

    """
    Function that develops the phenotype by max_steps steps from its initial state

    args:
    - W: matrix
    - C: initial genotype activation
    - max_steps: number of steps
    - activation: activation function for the function

    returns: 
    - phenotype: stable state (or nothing if not found)
    - steps: number of steps to find a stable state (or nothing if not found)
    """

    function develop(W, C, max_steps, activation)
        Pi = C
        Pf = activation(W * Pi)
        for i in 1:max_steps
            if Pi == Pf
                return Pf, i
            end
            Pi = Pf
            Pf = activation(W * Pi)
        end
        return nothing, nothing
    end

    """
    Applies multiplicative noise to selected nonzero elements in W. 

    - `W`: matrix to modify
    - `noise_prob`: chance to affect each nonzero element
    - `noise_dist`: multiplicative noise distribution (E[η] ≈ 1)
    """
    function apply_noise!(W::Matrix{Float64}, noise_prob::Float64, noise_dist::Distribution)
        for idx in findall(!iszero, W)
            if rand() < noise_prob
                W[idx] *= rand(noise_dist)
            end
        end
    end

    """
    Development with recurrent noise application to the matrix.
    * Noise applied every k-steps in the evolutionary path

    Arguments:
    - `W`: original regulation matrix
    - `C`: initial state vector
    - `max_steps`: max number of development steps
    - `activation`: activation function
    - `noise_prob`: fraction of nonzero weights affected each time
    - `noise_dist`: distribution for multiplicative noise (E[η]=1)
    - `k_noise`: number of steps between noise applications

    Returns:
    - final state (or nothing), number of steps (or nothing)
    """
    function develop_noise(W::Matrix{Float64},
        C::Vector{Float64},
        max_steps::Int,
        activation::Function;
        noise_prob::Float64=0.1,
        noise_dist::Distribution=Gamma(2, 1/2), #α=2, λ=2=θ^-1 
        k_noise::Int=10)

        W_current = copy(W)
        Pi = C
        Pf = activation(W_current * Pi)

        for step in 1:max_steps
            if Pi == Pf
                return Pf, step
            end

            Pi = Pf
            if step % k_noise == 0
                apply_noise!(W_current, noise_prob, noise_dist)
            end
            Pf = activation(W_current * Pi)
        end

        return nothing, nothing
    end


    """
    --------------------
    INITIALIZATION
    --------------------
    """

    """
    make_activity: A function that determines the initial state
    of the genes

    args:
    - N: number of genes per network
    """

    function make_activity(N::Int)
        return [(-1.)^i for i in 0:N-1]
    end

    """
    make_initial_states :A function that generates a list of pop_size vectors of
    length N corresponding to the initial vectors C

    args:
    - N: 
    """

    function make_initial_states(N::Int, pop_size::Int)
        activity = Matrix{Float64}(undef, pop_size, N)
        for i in 1:pop_size
            activity[i, :] .= make_activity(N)
        end
        return activity
    end

    """
    make_optimal_phenotype: Generate a vector of 1 and -1s

    args:
    - N_target: number of genes determining the phenotype
    - p: probability of elements being 1
    """

    function make_optimal_phenotype(N_target::Int, p::Float64)
        return sample([1, -1], Weights([p, 1 - p]), N_target)
    end


    """
    A function that generates an list of matrices according to some rules

    args:
    - mode: what kind of matrices will be part of the initial population
    - params: model parameters
    - intial_genotypes: Vector of vectors containing the vector C of each matrix
    - activation: activation function 

    returns:
    - matrices: Vector of matrices
    - new_phenotype: nothing if there is no need to update the optimal phenotype
    """

    function generate_initial_matrices(mode, params, initial_states, activation)

        pop_size = params["pop_size"]
        N_target = params["N_target"]
        N_regulator = params["N_regulator"]
        N = N_target + N_regulator
        mr = params["mr"]
        σr = params["σr"]  
        d = Normal(mr, σr)
        c = params["c"]

        phenotype = nothing

        if mode == "sparse"
            matrices = Vector{Any}(undef, pop_size)

            for i in 1:pop_size
                matrices[i] = rand(d, (N,N)) .* sample([0,1], Weights([1-c,c]), (N,N))
            end
            return (matrices, nothing)

        elseif  mode == "sparse,stable"
            max_steps = params["max_steps"]
            matrices = Vector{Any}(undef, pop_size)
            for i in 1:pop_size
                stable = false
                C = initial_states[i,:]
                while !stable 
                    candidate = rand(d,(N,N)) .* sample([0,1], Weights([1-c,c]), (N,N))
                    phenotype, steps = develop(candidate, C, max_steps, activation) 

                    # Non-noisy initialization?

                    if phenotype !== nothing  # if the matrix is table
                        matrices[i] = copy(candidate)
                        stable = true
                    end
                end
            end
            return (matrices, nothing)

        elseif mode == "sparse,unstable"
            max_steps = params["max_steps"]
            matrices = Vector{Any}(undef, pop_size)
            for i in 1:pop_size
                C = initial_states[i,:]
                not_stable = false
                while !not_stable 
                    candidate = rand(d,(N,N)) .* sample([0,1], Weights([1-c,c]), (N,N))
                    phenotype, steps = develop(candidate, C, max_steps, activation)
                    # Non-noisy initialization?
                    if phenotype === nothing  # if the matrix is unstable
                        matrices[i] = copy(candidate)
                        not_stable = true
                    end
                end
            end
            return (matrices, nothing)

        elseif mode == "sparse,founder"
            max_steps = params["max_steps"]
            matrices = Vector{Any}(undef, pop_size)
            stable = false
            while !stable
                C = initial_states[1,:]
                candidate = rand(d,(N,N)) .* sample([0,1], Weights([1-c,c]), (N,N))
                phenotype, steps = develop(candidate, C, max_steps, activation) 
                # Non-noisy initialization
                if phenotype !== nothing  # if the matrix is table
                    stable = true
                    for i in 1:pop_size
                        matrices[i] = copy(candidate)
                    end
                end
            end
            return (matrices, phenotype)
        
        else
            error("Unknown mode: $mode")
        end 
    end

    """
    -----------------
    STRUCTURES
    -----------------
    """

    """
    artificial_org: Individual in the population
    (Mutable)

    Properties:
    - N: Number of genes
    - W: Weighted, directed matrix representing the GRN
    - M: Unweighted, directed connectivity matrix 
    """

    @with_kw mutable struct artificial_org
        N::Int64 = 10;
        W::Matrix{Float64} = zeros((10,10));
        M::Matrix{Int} = zeros((10,10));
    end

    """
    artificial_pop: The population made of individuals of artificial_org
    (Mutable)

    Properties:
    - pop_size: number of organisms
    - N_regulator: number of regulator genes
    - N_target: number of target genes
    - phenotypic_optima: genotype expression (1,-1) configuration that is optimal
    - initial_states: initial genotype expression
    - pop_ens: list of organisms
    """

    @with_kw mutable struct artificial_pop
        pop_size::Int64;
        N_regulator::Int64 = 5;
        N_target::Int64 = 5;
        phenotypic_optima::Vector{Float64} = ones(5);
        initial_states:: Matrix{Float64} = ones((pop_size, 10));
        pop_ens = [artificial_org(N = N_regulator + N_target) for i in 1:pop_size];
    end

    """
    replace_matrices!: replaces (in-place) the weighted matrix of the organisms in a 
    population given a vector of matrices

    args:
    - pop: artificial_pop structure with the population
    - matrices: vector of matrices to replace the current weighted matrices
    """

    function replace_matrices!(pop::artificial_pop, matrices)
        for (organism,mat) in zip(pop.pop_ens,matrices)
            organism.W = mat
        end
    end 
    """
    ---------------------------
    MUTATION AND RECOMBINATION
    --------------------------
    """

    """
    reg_mutation!: Function that modifies W for a mutation on a nonzero entry
    using a resample method

    args: 
    - W: Weighted adjacency matrix
    - mr: mean of the normal distribution to resample
    - σr: std of the normal distribution to resample
    - pr: mutation probability
    """

    function reg_mutation!(W, mr, σr, pr)
        # Find all indices where W is non-zero
        nz_inds = findall(!iszero, W)
        
        if isempty(nz_inds)
            # @warn "No non-zero entries to mutate." 
            # Avoid verbose for small density parameters
            return
        end

        # Sample one of the non-zero positions
        # with pr chance
        if rand() < pr
            rand_ind = rand(nz_inds)    
            d = Normal(mr, σr)
            W[rand_ind] = rand(d)
        end 
    end
    """
    con_mutation!: a function that mutates the connectivity
    of a GRN by resampling the weights
    ** NOTE: It currently modifies a link at random,
    but we would like to experiment with different removal
    and addition mechanisms. We could have a probability of
    modifying an edge, and then have different rules and
    parameters for the addition and removal after a coin flip

    args:
    - W: weighted matrix
    - pc: probability of edge modification
    - mr, σr: distribution parameters of weights
    """
    function con_mutation!(W,pc,mr,σr)
        if rand() < pc # mutation happens
            m,n = size(W)
            i,j = rand(1:m,2)

            if W[i,j] != 0 # Turn off the edge
                W[i,j] = 0
            else # Turn on the edge
                d = Normal(mr, σr)
                W[i,j] = rand(d)
            end
        end
    end
    """
    individual_fitness: Evaluates the fitness of a single matrix 

    args: 
    - expressed_phenotype: expressed stable state of a matrix
    - optimal_phenotype: optimal phenotype
    - N_target: number of target genes
    - s: selection strength
    - distance: A function that takes the distance between two vectors 
        with elements 1,-1
    - unstable_fitness: Value of the fitness for unstable states
    """
    function indiv_fitness(expressed_phenotype, optimal_phenotype, N_target,s, distance, unstable_fitness)
        if expressed_phenotype !== nothing
            dist = distance(expressed_phenotype,optimal_phenotype,N_target)
            ws = exp(-s*dist)
            return ws
        else 
            return unstable_fitness
        end
    end

    """
    recombine_rows: A function that randomly swaps the rows of two matrices

    args:
    - A, B: matrices with the same dimensions.

    returns:
    - A matrix where each row comes from either A or B.
    """
    function recombine_rows(A::AbstractMatrix, B::AbstractMatrix, p_rec)
        @assert size(A) == size(B) "Matrices must have the same size"
        m, n = size(A)
        C = similar(A)
        
        if iszero(p_rec) # skip if there is no recombination
            return C
        end

        for i in 1:m
            if rand() > p_rec  # randomly true or false
                C[i, :] = A[i, :]
            else
                C[i, :] = B[i, :]
            end
        end
        return C
    end

    """
    create_offspring: creates pop_size matrices from an artificial_pop
    with a survival probability equal to their fitness from randomly sampled
    parents

    args:
    - pop: artificial_pop structure
        - pop_size: number of organisms
        - N_regulator: number of regulator genes
        - N_target: number of target genes
        - pop_ens: list of organisms
        - phenotypic_optima: optimal genotype expression
        - initial_states: initial genotype expression
    - activation: activation function 
    - distance: distance function
    - max_steps: maximum number of steps to find a stable state
    - s: selection strength
    - mr, σr: mutation distribution parameters (mean, std)
    - pr: mutation probability
    - unstable_fitness: default fitness for unstable matrices

    returns:
    - offspring: list of matrices
    - fitness: vector of fitnesses of each offspring matrix
    - steps: number of steps taken by each matrix for finding a stable state 
        * nothing if unstable
    - completion_gen: number of stable matrices in this generation
    - p_rec: probability of recombination
    ** NOTE: Other measures to be added in time series data
    """

    function create_offspring(pop::artificial_pop, activation,distance, max_steps, s, mr, σr, pr, unstable_fitness, p_rec,pc, noise_prob, noise_dist)

        pop_size = pop.pop_size
        phenotypic_optima = pop.phenotypic_optima
        initial_states = pop.initial_states
        pop_ens = pop.pop_ens
        N_target = pop.N_target
        N_genes = pop.N_target + pop.N_regulator

        survival = false 

        # Store offspring matrices
        offspring = Vector{Matrix{Float64}}(undef, pop_size)

        # Measures
        fitness = Vector{Float64}(undef,pop_size)
        completion_gen = 0
        steps = Vector{Any}(undef,pop_size)
        parents = Matrix{Int}(undef,pop_size, 2)
        noisy_W = Matrix{Float64}(undef, N_genes, N_genes)

        for i in 1:pop_size
            survival = false 
            while !survival 
                parent_i, parent_j = rand(1:pop_size, 2)
                # recombine
                W_candidate = recombine_rows(pop_ens[parent_i].W, pop_ens[parent_j].W, p_rec)
                
                # mutate
                reg_mutation!(W_candidate,mr,σr,pr)
                
                # Mutate connectivity of W_candidate
                con_mutation!(W_candidate,pc,mr,σr)

                # Make noise
                copyto!(noisy_W,W_candidate)
                apply_noise!(noisy_W,noise_prob,noise_dist)

                # find stable state
                C = initial_states[i,:]
                # phenotype, path_length = develop(W_candidate, C, max_steps, activation)
                phenotype, path_length = develop(noisy_W, C, max_steps, activation)

                # compute fitness
                fit = indiv_fitness(phenotype, phenotypic_optima, N_target, s, distance, unstable_fitness)
                
                # decide if the offspring survives
                if rand() < fit
                    offspring[i] = W_candidate
                    fitness[i] = fit
                    steps[i] = path_length
                    if phenotype !== nothing
                        completion_gen += 1
                    end
                    survival = true
                    parents[i,:] .= (parent_i,parent_j)
                end
            end 
        end 
        return offspring, fitness, steps, completion_gen, parents
    end

    """
    -----------------------
    EVOLUTIONARY ALGORITHM
    -----------------------
    """

    """
    run_simulation: a function that runs Wagner's (1996) evolutionary algorithm

    args:
    - parameters: simulation parameters in a Dict
        - G: Number of generations
        - pop_size: population size
        - N_target: number of target genes
        - N_regulator: number of regulator genes
        - c: connectivity of matrices
        - p: probability of 1 in optimal phenotype
        - mr, σr: mean and standard deviation of weights in adjacency matrices
        - pr: probability of mutation in matrix
        - s: selection strength
        - unstable_fitness: default fitness for unstable matrices
        - mode: mode of initial population:
            "sparse", "sparse,stable", "sparse,unstable", "sparse,founder"
    """
    function run_simulation(parameters)

        # PARAMETER ASSIGNATION
        G = parameters["G"]
        pop_size = parameters["pop_size"]
        N_target = parameters["N_target"]
        N_regulator = parameters["N_regulator"]
        N_genes = N_target + N_regulator
        c = parameters["c"]
        p = parameters["p"]
        mr = parameters["mr"]
        σr = parameters["σr"]
        pr = parameters["pr"]
        pc = parameters["pc"]
        max_steps = parameters["max_steps"]
        s = parameters["s"]
        p_rec = parameters["p_rec"]
        unstable_fitness = parameters["unstable_fitness"]
        mode = parameters["mode"]

        noise_prob = parameters["noise_prob"]
        noise_dist = parameters["noise_dist"]

        # INITIALIZATION
        initial_states = make_initial_states(N_target + N_regulator, pop_size)
        phenotypic_optima = make_optimal_phenotype(N_target, p)
        
        population = artificial_pop(pop_size = pop_size, N_regulator = N_regulator, N_target = N_target, phenotypic_optima = phenotypic_optima, initial_states = initial_states)
        initial_matrices, new_optima = generate_initial_matrices(mode, parameters, initial_states, activation)
        replace_matrices!(population, initial_matrices)

        # Update if there is a new optima generate from mode of initialziation
        if new_optima !== nothing
            population.phenotypic_optima = new_optima
            phenotypic_optima = new_optima
        end
        
        # MEASUREMENTS DECLARATIONS

        ## __ individual measures __ 
        fitness_history = Matrix{Float64}(undef, G+1,pop_size)
        path_length_history = Matrix{Any}(undef, G+1,pop_size)
        matrices_history = Array{Matrix{Float64}}(undef, G+1, pop_size)

        ## __ aggregate measures __
        completion = zeros(G+1)
        genealogy_tree = Array{Int}(undef, G, pop_size, 2)
        
        noisy_W = Matrix{Float64}(undef, N_genes, N_genes) # preallocate memory

        # Initial population measurements
        for (index, organism) in enumerate(population.pop_ens)

            copyto!(noisy_W,organism.W) # copy contents
            apply_noise!(noisy_W, noise_prob, noise_dist)

            # phenotype,path_length = develop(organism.W, population.initial_states[index, :], max_steps, activation)

            # noisy development in first generation
            # non-noisy development analyzed seperately
            phenotype, path_length = develop(noisy_W, population.initial_states[index, :], max_steps, activation)

            fit = indiv_fitness(phenotype, phenotypic_optima, N_target, s, distance, unstable_fitness)
            
            fitness_history[1,index] = fit
            if phenotype !== nothing # how many stable phenotypes there are
                completion[1] += 1 / pop_size
            end
            path_length_history[1,index] = path_length
            matrices_history[1,index] = organism.W
        end

        # RUN SIMULATION
        for gen in 2:G+1
            completion_gen = 0
            # compute the next generation (recombination, mutation, and fitness survival are implicit)
            offspring, fit, steps, completion_gen, parents = create_offspring(population, activation, distance, max_steps, s, mr, σr, pr, unstable_fitness, p_rec,pc,noise_prob,noise_dist)
            
            # store historic measures

            matrices_history[gen,:] .= offspring
            fitness_history[gen,:] .= fit
            path_length_history[gen,:] .= steps
            completion[gen] = completion_gen / pop_size
            genealogy_tree[gen-1,:,:] .= parents 

            # update matrices
            replace_matrices!(population, offspring)
        end

        # PROCESS DATA

        # currently, just group all of them together :)

        data = Dict("matrices"  => matrices_history,
                    "fitness" => fitness_history,
                    "path_length" => path_length_history,
                    "completion" => completion,
                    "genealogy_tree" => genealogy_tree,
                    "phenotypic_optima" => population.phenotypic_optima)


        return data
    end
end