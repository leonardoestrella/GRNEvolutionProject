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
    --------------------
    INITIALIZATION
    --------------------
    """

    """
    make_activity(N::Int) -> Vector{Float64}

    Generates the initial state of the genes for a network.

    args:
    - `N::Int`: Number of genes in the network.

    returns:
    - `Vector{Float64}`: A vector of length `N` with alternating values `1.0` and `-1.0`.
    """
    function make_activity(N::Int)
        return [(-1.)^i for i in 0:N-1]
    end
    
    """
    make_initial_states(N::Int, pop_size::Int) -> Matrix{Float64}

    Generates a matrix of initial gene activity states for a population of networks.

    args:
    - `N::Int`: Number of genes per network.
    - `pop_size::Int`: Number of networks in the population.

    returns:
    - `Matrix{Float64}`: A `pop_size × N` matrix where each row contains the initial gene states.
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
    generate_initial_matrices(params::Dict, initial_states::AbstractMatrix, activation) 
        -> Tuple{Vector{Any}, Any}

    Generates an initial population of gene interaction matrices according to the specified `mode`.

    Each matrix is `N × N`, where `N = N_target + N_regulator`, with entries drawn from a normal 
    distribution `Normal(mr, σr)` and sparsified by random masking with probability `c`.

    args
    - `params::Dict`: Dictionary of parameters. Must include:
        - `mode::String`: Determines the generation strategy:
            - `"sparse"`: Random sparse matrices with no stability check.
            - `"sparse,stable"`: Random sparse matrices that yield a stable phenotype under `develop`.
            - `"sparse,unstable"`: Random sparse matrices that yield an unstable phenotype under `develop`.
            - `"sparse,founder"`: A single stable matrix is found and duplicated for the entire population.
        - `"pop_size"`: Number of matrices to generate.
        - `"N_target"`: Number of target genes.
        - `"N_regulator"`: Number of regulator genes.
        - `"mr"`: Mean of the normal distribution for weights.
        - `"σr"`: Standard deviation of the normal distribution for weights.
        - `"c"`: Connection probability (controls sparsity).
        - `"max_steps"`: Maximum number of steps for phenotype development (used in stable/unstable/founder modes).
    - `initial_states::AbstractMatrix`: A `pop_size × N` matrix of initial gene states.
    - `activation`: Activation function passed to `develop`.

    returns
    - `Tuple`:
        1. `Vector{Any}`: The generated matrices.
        2. `Any`: The phenotype of the founder matrix (only in `"sparse,founder"` mode; otherwise `nothing`).

    # Notes
    - Stability is determined by the `develop` function, which returns `(phenotype, steps)`.
    - A phenotype is considered stable if it is not `nothing`.
    """

    function generate_initial_matrices(params, initial_states, activation)
        
        mode = params["mode"]
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
    """

    @with_kw mutable struct artificial_org
        N::Int64 = 10;
        W::Matrix{Float64} = zeros((10,10));
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
    of a GRN by adding or removing a weight. 
    ** NOTE: We would like to experiment with different removal
    and addition mechanisms. 

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
    create_offspring(pop::artificial_pop, activation, distance, params::Dict) 
        -> Tuple{
            Vector{Matrix{Float64}},  # offspring
            Vector{Float64},          # fitness
            Vector{Any},              # steps
            Int,                      # completion_gen
            Matrix{Int}               # parents
        }

    Generates a new generation of offspring matrices from an existing `artificial_pop`
    using recombination, mutation, and fitness-based selection.

    # Arguments
    - `pop::artificial_pop`: Population containing individuals and their properties.
    - `activation`: Activation function used during development.
    - `distance`: Distance metric for computing fitness.
    - `params::Dict`: Dictionary with the following keys:
        - `"s"::Float64`: Selection strength.
        - `"mr"::Float64`: Mutation rate for regulatory weights.
        - `"σr"::Float64`: Standard deviation of weight mutations.
        - `"pr"::Float64`: Probability of regulatory weight mutation.
        - `"unstable_fitness"::Float64`: Fitness assigned to unstable phenotypes.
        - `"p_rec"::Float64`: Probability of recombination per row.
        - `"pc"::Float64`: Probability of connectivity mutation.
        - `"noise_prob"::Float64`: Probability of noise applied to weights.
        - `"noise_dist"`: Distribution from which noise is drawn.
        - `"max_steps"::Int`: Maximum number of steps to attempt reaching a stable state.

    # Returns
    A tuple containing:
    1. `Vector{Matrix{Float64}}`: Offspring weight matrices.
    2. `Vector{Float64}`: Fitness of each offspring.
    3. `Vector{Any}`: Number of steps each offspring took to reach a stable state (`nothing` if unstable).
    4. `Int`: Number of offspring that reached stability (`completion_gen`).
    5. `Matrix{Int}`: Parent indices for each offspring (`pop_size × 2`).

    # Notes
    - Offspring are accepted into the new generation with a probability equal to their computed fitness.
    - Stability is determined by the `develop` function; a stable phenotype is any non-`nothing` return value.
    - Noise is applied after mutation but before development.
    """

    function create_offspring(pop::artificial_pop, activation,distance, params)

        s = params["s"]
        mr = params["mr"]
        σr = params["σr"]
        pr = params["pr"]
        unstable_fitness = params["unstable_fitness"]
        p_rec = params["p_rec"]
        pc = params["pc"]
        noise_prob = params["noise_prob"]
        noise_dist = params["noise_dist"]
        max_steps = params["max_steps"]

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
    run_simulation(parameters::Dict) -> Dict

    Runs an evolutionary algorithm based on and modified from Wagner (1996) for a given number of generations,
    starting from an initialized population of gene regulatory networks, and records
    population- and individual-level measurements over time.

    # Arguments
    - `parameters::Dict`: Simulation parameters, containing:
        - `"G"::Int`: Number of generations.
        - `"pop_size"::Int`: Population size.
        - `"N_target"::Int`: Number of target genes.
        - `"N_regulator"::Int`: Number of regulator genes.
        - `"c"`: Connectivity of matrices.
        - `"p"::Float64`: Probability of 1 in optimal phenotype.
        - `"mr"::Float64`: Mutation rate for regulatory weights.
        - `"σr"::Float64`: Standard deviation of weight mutations.
        - `"pr"::Float64`: Probability of regulatory weight mutation.
        - `"s"::Float64`: Selection strength.
        - `"unstable_fitness"::Float64`: Fitness assigned to unstable phenotypes.
        - `"mode"::String`: Initialization mode (`"sparse"`, `"sparse,stable"`, `"sparse,unstable"`, `"sparse,founder"`).
        - `"max_steps"::Int`: Maximum number of steps to reach a stable state.
        - `"noise_prob"::Float64`: Probability of noise applied to weights.
        - `"noise_dist"`: Distribution from which noise is drawn.
        - Any other keys required by `generate_initial_matrices` and `create_offspring`.

    # Returns
    `Dict` containing:
    - `"matrices"`: `Array{Matrix{Float64}}` of shape `(G+1, pop_size)`, weight matrices for each generation and individual.
    - `"fitness"`: `Matrix{Float64}` of shape `(G+1, pop_size)`, fitness values per generation and individual.
    - `"path_length"`: `Matrix{Any}` of shape `(G+1, pop_size)`, number of steps to stability (or `nothing` if unstable).
    - `"completion"`: `Vector{Float64}` of length `G+1`, fraction of stable individuals per generation.
    - `"genealogy_tree"`: `Array{Int}` of shape `(G, pop_size, 2)`, parent indices for each offspring.
    - `"phenotypic_optima"`: Final optimal phenotype.

    # Notes
    - The first generation is evaluated with noise applied before development.
    - Stability is determined by the `develop` function; unstable phenotypes return `nothing`.
    - Offspring generation (recombination, mutation, survival) is handled by `create_offspring`.
    """

    function run_simulation(parameters)

        # PARAMETER ASSIGNATION
        G = parameters["G"]
        pop_size = parameters["pop_size"]
        N_target = parameters["N_target"]
        N_regulator = parameters["N_regulator"]
        N_genes = N_target + N_regulator
        p = parameters["p"]
        max_steps = parameters["max_steps"]
        s = parameters["s"]
        unstable_fitness = parameters["unstable_fitness"]

        noise_prob = parameters["noise_prob"]
        noise_dist = parameters["noise_dist"]

        # INITIALIZATION
        initial_states = make_initial_states(N_target + N_regulator, pop_size)
        phenotypic_optima = make_optimal_phenotype(N_target, p)
        
        population = artificial_pop(pop_size = pop_size, N_regulator = N_regulator, N_target = N_target, phenotypic_optima = phenotypic_optima, initial_states = initial_states)
        initial_matrices, new_optima = generate_initial_matrices(parameters, initial_states, activation)
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
            offspring, fit, steps, completion_gen, parents = create_offspring(population, activation, distance, parameters)
            
            # store historic measures

            matrices_history[gen,:] .= offspring
            fitness_history[gen,:] .= fit
            path_length_history[gen,:] .= steps
            completion[gen] = completion_gen / pop_size
            genealogy_tree[gen-1,:,:] .= parents 

            # update matrices
            replace_matrices!(population, offspring)
        end

        data = Dict("matrices"  => matrices_history,
                    "fitness" => fitness_history,
                    "path_length" => path_length_history,
                    "completion" => completion,
                    "genealogy_tree" => genealogy_tree,
                    "phenotypic_optima" => population.phenotypic_optima)

        return data
    end
end