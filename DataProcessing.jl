module CustomStats

    using Plots, Statistics, Distributions

    include("WagnerAlgorithm.jl")
    import .BooleanNetwork: reg_mutation!, con_mutation!, develop

export rowwise_summary, compute_densities, fit_path_stable_plot, compare_distributions, summarize_experiment_data, mut_robustness_population, path_mat_analysis_plot, noise_comparison_timeseries_plot
    
    """
    ---------------------
    DATA PROCESSING
    -----------------------
    """

    """
    ....................
    Time-series processing
    ......................
    """

    """
    rowwise_summary(data) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

    Computes summary statistics (mean, median, and standard deviation) for each row of a 2D array,
    ignoring `nothing` values.

    # Arguments
    - `data`: A 2D array or matrix where each row contains numeric values or `nothing`.

    # Returns
    A tuple containing:
    1. `Vector{Float64}`: Row means.
    2. `Vector{Float64}`: Row medians.
    3. `Vector{Float64}`: Row standard deviations.

    # Notes
    - Rows containing only `nothing` are skipped (no entry is added to the results).
    - All `nothing` values are excluded from the computation of statistics.
    """

    function rowwise_summary(data)
        means   = Float64[]
        medians = Float64[]
        stds    = Float64[]

        for i in axes(data, 1)
            row = data[i, :]
            valid_values = filter(!isnothing, row)

            if length(valid_values) > 0
                push!(means,   mean(valid_values))
                push!(medians, median(valid_values))
                push!(stds,    std(valid_values))
            end
        end

        return means, medians, stds
    end

    """
    ...................
    Mutations
    ...................
    """

    """
    Functions included in BooleanNetwork module:

    reg_mutation!(W, mr, σr, pr): Function that modifies W for a mutation on a nonzero entry
    using a resample method

        args: 
        - W: Weighted adjacency matrix
        - mr: mean of the normal distribution to resample
        - σr: std of the normal distribution to resample
        - pr: mutation probability

    con_mutation!(W,pc,mr,σr): a function that mutates the connectivity
    of a GRN by resampling the weights

        args:
        - W: weighted matrix
        - pc: probability of edge modification
        - mr, σr: distribution parameters of weights

    develop(W, C, max_steps, activation) function that develops the phenotype by max_steps steps from its initial state

        args:
        - W: matrix
        - C: initial genotype activation
        - max_steps: number of steps
        - activation: activation function for the function

        returns: 
        - phenotype: stable state (or nothing if not found)
        - steps: number of steps to find a stable state (or nothing if not found)
    """

    """
    mut_robustness_population(
        Ws::Vector{Matrix{Float64}},
        mr::Float64,
        σr::Float64,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int
    ) -> Vector{Float64}

    Estimates the mutational robustness of a population of weight matrices by repeatedly mutating 
    nonzero elements and measuring whether the resulting phenotype remains unchanged.

    # Arguments
    - `Ws`: Vector of adjacency/weight matrices representing individuals in the population.
    - `mr::Float64`: Mean of the normal distribution used for mutation values.
    - `σr::Float64`: Standard deviation of the normal distribution used for mutation values.
    - `initial_states`: Matrix where each row is the initial state vector for the corresponding individual.
    - `max_steps::Int`: Maximum number of steps allowed for phenotype development.
    - `activation::Function`: Activation function used during development.
    - `trials::Int`: Number of random mutations performed per individual.

    # Returns
    - `Vector{Float64}`: Robustness score for each matrix, defined as the fraction of mutations 
    that do not change the original phenotype.

    # Notes
    - Only nonzero weights in each matrix are considered for mutation.
    - If the original phenotype is unstable (`nothing`), robustness is set to `0.0`.
    - The original phenotype for each matrix is computed before any mutations.
    - Mutations are temporary and the matrix is restored after each trial.
    - Uses multi-threading for faster computation across the population.
    """

    function mut_robustness_population(Ws::Vector{Matrix{Float64}},
        mr::Float64,
        σr::Float64,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int)

        n = length(Ws)
        robustness_scores = zeros(Float64, n)
        d = Normal(mr, σr)

        # Precompute original phenotypes
        orig_phenotypes = Vector{Union{Vector{Float64},Nothing}}(undef, n)
        for i in 1:n
            phen, _ = develop(Ws[i], initial_states[i, :], max_steps, activation)
            orig_phenotypes[i] = phen
        end

        Threads.@threads for i in 1:n
            W = Ws[i]
            state = initial_states[i, :]
            orig_phen = orig_phenotypes[i]

            if orig_phen === nothing
                continue
            end

            # Preselect non-zero indices
            nz_inds = findall(!iszero, W)
            if isempty(nz_inds)
                continue
            end

            sampled_inds = rand(nz_inds, trials)
            sampled_vals = rand(d, trials)

            invariant_count = 0
            for t in 1:trials
                idx = sampled_inds[t]
                old_val = W[idx]
                W[idx] = sampled_vals[t]

                phen, _ = develop(W, state, max_steps, activation)

                if phen !== nothing && phen == orig_phen
                    invariant_count += 1
                end

                W[idx] = old_val  # Restore original weight
            end

            robustness_scores[i] = invariant_count / trials
        end

        return robustness_scores
    end

    """
    noise_robustness_population(
        Ws::Vector{Matrix{Float64}},
        noise_dist::Distribution,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int
    ) -> Vector{Float64}

    Estimates the noise robustness of a population of weight matrices by repeatedly applying
    multiplicative noise to nonzero elements and measuring whether the phenotype remains unchanged.

    # Arguments
    - `Ws`: Vector of adjacency/weight matrices representing individuals in the population.
    - `noise_dist::Distribution`: Distribution from which multiplicative noise factors are sampled.
    - `initial_states`: Matrix where each row is the initial state vector for the corresponding individual.
    - `max_steps::Int`: Maximum number of steps allowed for phenotype development.
    - `activation::Function`: Activation function used during development.
    - `trials::Int`: Number of noise perturbations tested per individual.

    # Returns
    - `Vector{Float64}`: Robustness score for each matrix, defined as the fraction of noise perturbations
    that do not change the original phenotype.

    # Notes
    - Only nonzero weights in each matrix are considered for perturbation.
    - If the original phenotype is unstable (`nothing`), robustness is set to `0.0`.
    - The original phenotype for each matrix is computed before any perturbations.
    - Noise is applied multiplicatively: `W[idx] = sampled_noise * W[idx]`.
    - Perturbations are temporary; the matrix is restored after each trial.
    - Uses multi-threading to process individuals in parallel.
    """

    function noise_robustness_population(Ws::Vector{Matrix{Float64}},
        noise_dist::Distribution,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int)

        n = length(Ws)
        robustness_scores = zeros(Float64, n)

        # Precompute noiseless phenotypes
        orig_phenotypes = Vector{Union{Vector{Float64},Nothing}}(undef, n)
        for i in 1:n
            phen, _ = develop(Ws[i], initial_states[i, :], max_steps, activation)
            orig_phenotypes[i] = phen
        end

        Threads.@threads for i in 1:n
            W = Ws[i]
            state = initial_states[i, :]
            orig_phen = orig_phenotypes[i]

            if orig_phen === nothing
                continue
            end

            # Preselect non-zero indices
            nz_inds = findall(!iszero, W)
            if isempty(nz_inds)
                continue
            end

            sampled_inds = rand(nz_inds, trials)
            sampled_vals = rand(noise_dist, trials)

            invariant_count = 0
            for t in 1:trials
                idx = sampled_inds[t]
                old_val = W[idx]
                W[idx] = sampled_vals[t] * W[idx]

                phen, _ = develop(W, state, max_steps, activation)

                if phen !== nothing && phen == orig_phen
                    invariant_count += 1
                end

                W[idx] = old_val  # Restore original weight
            end

            robustness_scores[i] = invariant_count / trials
        end

        return robustness_scores
    end

    """
    compute_noiseless_dynamics: Simulates noiseless Boolean network development over evolutionary time.

    # Arguments
    - `evol_matrices::Array`: A `G+1 × pop_size` matrix of evolved Boolean network matrices.
    - `phen_opt::Vector`: Phenotypic optima for the population.
    - `parameters::Dict`: Dictionary containing all necessary parameters:
        - `"G"`: Number of generations.
        - `"pop_size"`: Number of individuals per generation.
        - `"N_target"`: Number of target genes.
        - `"N_regulator"`: Number of regulator genes.
        - `"max_steps"`: Maximum steps allowed in development.
        - `"s"`: Selection pressure.
        - `"unstable_fitness"`: Fitness value for unstable individuals.

    # Returns
    A named tuple:
    - `fit::Matrix{Float64}`: Fitness values for all individuals across generations.
    - `path::Matrix{Any}`: Developmental paths (number of steps) for each individual.
    - `completion::Vector{Float64}`: Proportion of individuals per generation that completed development.
    """
    function compute_noiseless_dynamics(evol_matrices::Matrix, phen_opt::Vector, parameters::Dict)

        G = parameters["G"]
        pop_size = parameters["pop_size"]
        N_genes = parameters["N_target"] + parameters["N_regulator"]

        # Initialize data containers
        fit = Matrix{Float64}(undef, G + 1, pop_size)
        path = Matrix{Any}(undef, G + 1, pop_size)
        completion = Vector{Float64}(undef, G + 1)

        # Precompute initial states
        initial_states = BooleanNetwork.make_initial_states(N_genes, pop_size)

        # Iterate over generations and individuals
        for gen in 1:G+1
            count_complete = 0

            for (idx, mat) in enumerate(evol_matrices[gen, :])
                phen, steps = BooleanNetwork.develop(
                    mat,
                    initial_states[idx, :],
                    parameters["max_steps"],
                    BooleanNetwork.activation
                )

                fit[gen, idx] = BooleanNetwork.indiv_fitness(
                    phen, phen_opt, parameters["N_target"], parameters["s"],
                    BooleanNetwork.distance, parameters["unstable_fitness"]
                )
                path[gen, idx] = steps

                if phen !== nothing
                    count_complete += 1
                end
            end

            completion[gen] = count_complete / pop_size
        end

        return (fit=fit, path=path, completion=completion)
    end

    """
    .............
    Matrix Analysis
    .............
    """

    """
    get_nonzero: a function that gets the nonzero elements of a matrix

    args:
    - weighted: a Matrix where to extract the non-zero elements from
    - positive: a Bool that indicates whether keeping the sign of the weights
        or just their presence (default to presence)

    returns:
    connectivity: a Matrix
    """

    function get_nonzero(weighted; positive = true)
        if positive
            return weighted .!= 0
        else
            return sign.(weighted)
        end
    end

    """
    compute_densities(data::Array{<:AbstractMatrix, 2}) -> Matrix{Float64}

    Computes the density of positive (nonzero) entries for each matrix in a 
    generations × population array.

    # Arguments
    - `data`: A 2D array of matrices with shape `(generations, population_size)`.

    # Returns
    - `Matrix{Float64}`: A matrix of the same shape as `data` where each entry 
    is the fraction of positive (nonzero) entries in the corresponding matrix.
    """
    function compute_densities(data::Array{<:AbstractMatrix,2})
        gens, pop_size = size(data)
        densities = Matrix{Float64}(undef, gens, pop_size)

        for i in axes(data, 1)
            densities[i, :] .= map(mat -> begin
                    adj = get_nonzero(mat, positive=true)
                    count(adj) / length(adj)
                end, data[i, :])
        end
        return densities
    end

    """
    ----------------
    SUMMARIZING FUNCTIONS
    -----------------
    """
    
    """
    summarize_experiment_data(data::Dict) -> Dict

    Computes summary statistics for key experimental data metrics, aggregating
    per-generation and per-individual measurements.

    # Arguments
    - `data::Dict`: Dictionary containing experiment results with at least the keys:
        - `"fitness"`: Matrix of fitness values (generations × population).
        - `"path_length"`: Matrix of path lengths (generations × population).
        - `"matrices"`: Array of weight matrices (generations × population).
        - `"completion"`: Vector of completion rates per generation.

    # Returns
    - `Dict`: Dictionary with the following entries:
        - `"avg_fit"`, `"median_fit"`, `"std_fit"`: Row-wise mean, median, and std of fitness.
        - `"avg_path"`, `"median_path"`, `"std_path"`: Row-wise mean, median, and std of path length.
        - `"completion"`: Passed through from input data.
        - `"avg_density"`, `"median_density"`, `"std_density"`: Row-wise mean, median, and std of matrix densities.

    # Notes
    - Uses `rowwise_summary` to compute statistics ignoring missing or `nothing` values.
    - Computes densities of positive entries in matrices before summarizing.
    """

    function summarize_experiment_data(data)

        processed_data = Dict()

        avg_fit, median_fit, std_fit = rowwise_summary(data["fitness"])
        avg_path, median_path, std_path = rowwise_summary(data["path_length"])

        densities = compute_densities(data["matrices"])
        avg_density, median_density, std_density = rowwise_summary(densities)

        processed_data["avg_fit"] = avg_fit
        processed_data["median_fit"] = median_fit
        processed_data["std_fit"] = std_fit

        processed_data["avg_path"] = avg_path
        processed_data["median_path"] = median_path
        processed_data["std_path"] = std_path

        processed_data["completion"] = data["completion"]

        processed_data["avg_density"] = avg_density
        processed_data["median_density"] = median_density
        processed_data["std_density"] = std_density

        return processed_data
    end
end