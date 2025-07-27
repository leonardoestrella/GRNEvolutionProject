module CustomStats

    using Plots, Statistics, Distributions

    include("WagnerAlgorithm.jl")
    import .BooleanNetwork: reg_mutation!, con_mutation!, develop

export rowwise_summary, get_densities, fit_path_stable_plot, compare_distributions, summarize_experiment_data, mut_robustness_population, path_mat_analysis_plot, noise_comparison_timeseries_plot

    """
    --------------------------
    PLOTTING
    -------------------------
    """

    """
    plot_timeseries_summary: Plot a summary of time series data using average, median, and standard deviation.

    args:
    - `avg`: Vector of average values over time.
    - `med`: Vector of median values over time.
    - `stddev`: Vector of standard deviations over time.

    kwargs:
    - `time`: Vector of time points (defaults to `1:length(avg)`).
    - `ylabel`: Label for the y-axis (default: "").
    - `xlabel`: Label for the x-axis (default: "").
    - `label_prefix`: String prefix added to legend labels (default: "").
    - `color`: Color for the average line and ribbon (default: `:blue`).
    - `plt`: Existing plot object to add to (optional).

    returns:
    - A plot object showing:
        - The average line with a ±2 standard deviation ribbon.
        - The median as a dashed black line.
    """

    function plot_timeseries_summary(avg,med,stddev;
                                    time=nothing,
                                    ylabel="",
                                    xlabel="",
                                    label_prefix="",
                                    color=:blue,
                                    plt=nothing)
        n = size(avg)[1]
        if isnothing(time)
                time = 1:n
        end

        # Use provided plot or start a new one
        plt = isnothing(plt) ? plot() : plt

        # Plot mean with ±2σ ribbon
        plot!(plt, time, avg, ribbon=2*stddev, fillalpha=0.2, lw=2,
                color=color, label="$label_prefix Mean ±2 SD")

        # Plot median as dashed line
        plot!(plt, time, med, linestyle=:dash, lw=2, color=:black, label="$label_prefix Median")
        plot!(plt, legend=:topright)
        # Label y-axis
        ylabel!(plt, ylabel)
        xlabel!(plt, xlabel)

        return plt
    end

    """
    fit_path_stable_plot: Generate a multi-panel plot summarizing evolutionary time series data, including fitness, path length, stability, and edge density.

    args:
    - `avg_fitnesses`: Vector of average fitness values over generations.
    - `std_fitnesses`: Vector of standard deviations of fitness values.
    - `avg_path`: Vector of average path lengths over generations.
    - `std_path`: Vector of standard deviations of path lengths.
    - `completion`: Vector of fractions (0–1) of stable matrices per generation.
    - `avg_density`: Vector of average edge densities over generations.
    - `std_density`: Vector of standard deviations of edge densities.
    - `safe_title`: A string used to construct the output filename (should be file-system safe).
    - `path`: Directory path where the figure will be saved.
    - `plot_title`: Title for the fitness plot.
    - `cpal`: A vector of color values used for plotting each series.

    kwargs:
    - `show_fig`: Whether to display the figure (default: `false`).

    behavior:
    Generates and saves a 4-row subplot figure including:
    1. **Fitness**: Mean ± 2σ with ribbon.
    2. **Path Length**: Mean ± 2σ with ribbon.
    3. **Stability**: Proportion of stable matrices as scatter points.
    4. **Edge Density**: Mean ± 2σ with ribbon.

    The figure is saved as a PNG to the given `path` with a filename based on `safe_title`.

    returns
    - Nothing. Saves and optionally displays the plot.
    """
    function fit_path_stable_plot(
        avg_fitnesses,std_fitnesses, avg_path, 
        std_path,completion,avg_density, std_density,
        safe_title,path,plot_title, cpal; 
        show_fig = false)

        generations = size(avg_fitnesses,1)

        p1 = plot(1:generations, avg_fitnesses, 
            ribbon = 2 .* std_fitnesses, seriestype = :line,
            markersize = 1, alpha = 0.7, title = plot_title,
            ylabel = "Fitness", legend = :topright,
            label = "Average Fitness ± 2σ",
            color = cpal[1], left_margin=6Plots.mm, top_margin = 5Plots.mm)

        p2 = plot(1:generations, avg_path, 
            ribbon = 2 .* std_path, seriestype = :line,
            markersize = 1, alpha = 0.7,
            ylabel = "Path Length", legend = :topright,
            label = "Average Path Length ± 2σ",
            color = cpal[3], left_margin=6Plots.mm)

        p3 = plot(1:generations, completion, 
            seriestype = :scatter, legend = :topright,
            markersize = 3, alpha = 0.7,
            ylabel = "% of Stable Matrices",
            color = cpal[5], ylims = (-0.1, 1.1), left_margin=6Plots.mm)

        p4 = plot(1:generations, avg_density, 
            ribbon = 2 .* std_density, seriestype = :line,
            markersize = 1, alpha = 0.7, xlabel = "Generation",
            ylabel = "Density of Edges", legend = :topright,
            label = "Average Density ± 2σ",
            color = cpal[7], left_margin=6Plots.mm, bottom_margin = 6Plots.mm)

        # Combine into a single layout
        layout = @layout([a; b; c; d])
        final_plot = plot(p1, p2, p3, p4; layout=layout, size=(800, 1200))

        savefig(final_plot, "$path$(safe_title)_timeseries1.png")
        # Display in notebook
        if show_fig
            display(final_plot)
        end
    end

    """
    A very similar plot to fit_path_stable_plot, but with a comparison between
    noisy and non-noisy development
    """
    function noise_comparison_timeseries_plot(
        avg_fitnesses, std_fitnesses, avg_path,
        std_path, completion, avg_density, std_density,
        avg_fit_noiseless, std_fit_noiseless,
        avg_path_noiseless, std_path_noiseless,
        noiseless_completion,
        safe_title, path, plot_title, cpal;
        show_fig=false)

        generations = size(avg_fitnesses, 1)

        p1 = plot(1:generations, avg_fitnesses,
            ribbon=2 .* std_fitnesses, seriestype=:line,
            markersize=1, alpha=0.6, title=plot_title,
            ylabel="Fitness", legend=:topright,
            label="Noisy Average Fitness ± 2σ",
            color=cpal[1], left_margin=6Plots.mm, top_margin=5Plots.mm)

        plot!(p1, 1:generations, avg_fit_noiseless,
            ribbon=2 .* std_fit_noiseless, seriestype=:line, lynestyle=:dash,
            markersize=1, alpha=0.6,
            ylabel="Fitness", legend=:topright,
            label="Noiseless Average Fitness ± 2σ",
            color=cpal[8], left_margin=6Plots.mm, top_margin=5Plots.mm)

        p2 = plot(1:generations, avg_path,
            ribbon=2 .* std_path, seriestype=:line,
            markersize=1, alpha=0.6,
            ylabel="Path Length", legend=:topright,
            label="Noisy Average Path Length ± 2σ",
            color=cpal[2], left_margin=6Plots.mm)

        plot!(p2, 1:generations, avg_path_noiseless,
            ribbon=2 .* std_path_noiseless, seriestype=:line, lynestyle=:dash,
            markersize=1, alpha=0.6,
            ylabel="Path Length", legend=:topright,
            label="Noiseless Average Path Length ± 2σ",
            color=cpal[10], left_margin=6Plots.mm, top_margin=5Plots.mm)

        p3 = plot(1:generations, completion,
            seriestype=:scatter, legend=:topright,
            markersize=3, alpha=0.6,
            ylabel="% of Stable Matrices",
            label="Noisy Development",
            color=cpal[3], ylims=(-0.1, 1.1), left_margin=6Plots.mm)

        plot!(p3, 1:generations, noiseless_completion,
            seriestype=:scatter, markershape=:utriangle,
            markersize=3, alpha=0.5,
            ylabel="Completion", legend=:topright,
            label="Noiseless Development",
            color=cpal[12], left_margin=6Plots.mm, top_margin=5Plots.mm)

        p4 = plot(1:generations, avg_density,
            ribbon=2 .* std_density, seriestype=:line,
            markersize=1, alpha=0.7, xlabel="Generation",
            ylabel="Density of Edges", legend=:topright,
            label="Average Density ± 2σ",
            color=cpal[7], left_margin=6Plots.mm, bottom_margin=6Plots.mm)

        # Combine into a single layout
        layout = @layout([a; b; c; d])
        final_plot = plot(p1, p2, p3, p4; layout=layout, size=(800, 1200))

        savefig(final_plot, "$path$(safe_title)_timeseries1.png")
        # Display in notebook
        if show_fig
            display(final_plot)
        end
    end

    """
    path_mat_analysis_plot: Generate a multi-panel plot summarizing evolutionary time series data, including fitness, path length, stability, and edge density.

    args:
    - `avg_rob`: Vector of average mutational robustness values over generations.
    - `std_rob`: Vector of standard deviations of mut robustness values.
    - `safe_title`: A string used to construct the output filename (should be file-system safe).
    - `path`: Directory path where the figure will be saved.
    - `plot_title`: Title for the fitness plot.
    - `cpal`: A vector of color values used for plotting each series.

    kwargs:
    - `show_fig`: Whether to display the figure (default: `false`).

    behavior:
    Generates and saves a 4-row subplot figure including:
    1. **Mutational Robustness**: Mean ± 2σ with ribbon.

    The figure is saved as a PNG to the given `path` with a filename based on `safe_title`.

    returns
    - Nothing. Saves and optionally displays the plot.
    """
    function path_mat_analysis_plot(
        avg_rob_mut,std_rob_mut,
        avg_rob_con, std_rob_con,
        safe_title,path,plot_title, cpal; 
        show_fig = false)

        generations = size(avg_rob_mut,1)
        p1 = plot(1:generations, avg_rob_mut, 
            ribbon = 2 .* std_rob_mut, seriestype = :line,
            markersize = 1, alpha = 0.7, title = plot_title,
            ylabel = "Mutational Robustness", legend = :topright,
            label = "Average Mutational Robustness ± 2σ",
            color = cpal[9], left_margin=6Plots.mm)
        p2 = plot(1:generations, avg_rob_con,
            ribbon=2 .* std_rob_con, seriestype=:line,
            markersize=1, alpha=0.7, title=plot_title,
            ylabel="Connectivity Robustness", xlabel="Generation", legend=:topright,
            label="Average Connectivity Robustness ± 2σ",
            color=cpal[13], left_margin=6Plots.mm, bottom_margin=6Plots.mm)

        # Combine into a single layout
        layout = @layout([a; b])
        final_plot = plot(p1, p2; layout=layout, size=(800, 600))
        
        savefig(final_plot, "$path$(safe_title)_timeseries2.png")
        # Display in notebook
        if show_fig
            display(final_plot)
        end
    end

    """
    compare_distributions: 
    Plot a histogram comparing two distributions

    args:
    - `distribution1`: Vector of initial path lengths (cannot include `nothing` for unstable cases).
    - `distribution2`: Vector of final path lengths (cannot include `nothing` for unstable cases).
    - `plot_title`: Title of the plot 
    - `safe_title`: A string used to construct the output filename (should be file-system safe).
    - `path`: Directory path where the plot image will be saved.
    - `cpal`: Color palette used for the histogram bars (expects at least two colors).

    kwargs
    - `show_fig`: Whether to display the figure after saving (default: `false`).

    behavior
    - Filters out `nothing` values (unstable cases) from both path length vectors.
    - Plots overlapping histograms of the initial and final path lengths, normalized as PDFs.
    - Saves the figure as `safe_title_histogram.png` in the given `path`.

    returns
    - Nothing. Saves and optionally displays the plot.
    """

    function compare_distributions(distribution1, distribution2,
                            plot_title, lab1, lab2,
                            safe_title, path,
                            cpal, num_bins; show_fig = false)

        # Filter out `nothing` values
        filtered1 = filter(!isnothing, distribution1)
        filtered2 = filter(!isnothing, distribution2)

        N1 = length(filtered1)
        N2 = length(filtered2)

        # Determine bin edges and xticks
        if N1 > 0 && N2 > 0
            m1, n1 = maximum(filtered1), minimum(filtered1)
            m2, n2 = maximum(filtered2), minimum(filtered2)
            max_bins = maximum([m1, m2])
            min_bins = minimum([n1, n2])
        elseif N1 > 0
            max_bins = maximum(filtered1)
            min_bins = minimum(filtered1)
        elseif N2 > 0
            max_bins = maximum(filtered2)
            min_bins = minimum(filtered2)
        else
            # Both empty — create dummy bins around 0
            min_bins = -0.5
            max_bins = 0.5
        end

        # Construct edges and ticks
        if max_bins == min_bins
            ε = typeof(max_bins) <: Int ? 0.5 : 1e-6
            edges = [min_bins - ε, min_bins + ε]
            xticks = [min_bins]
        else
            if typeof(max_bins) <: Integer
                edges = min_bins-0.5:1:max_bins+0.5
                xticks = floor(Int, minimum(edges)):ceil(Int, maximum(edges))
            else
                edges = range(min_bins, max_bins; length=num_bins + 1)
                xticks = range(min_bins, max_bins; length=min(num_bins, 10))
            end
        end

        tick_labels = string.(round.(xticks; digits=2))

        # Start plot
        if N1 == 0 && N2 == 0
            histogram([0]; bins=edges,
                alpha = 0.0,
                label = "Both distributions are empty",
                xlabel = "Value",
                ylabel = "Frequency",
                title = plot_title,
                legend = :topright,
                normalize = :pdf,
                color = :gray,
                xticks = (xticks, tick_labels),
                margin = 6Plots.mm
            )
        else
            if N1 > 0
                histogram(filtered1;
                    bins = edges,
                    alpha = 0.5,
                    label = "$lab1. N = $N1",
                    xlabel = "Value",
                    ylabel = "Frequency",
                    title = plot_title,
                    legend = :topright,
                    normalize = :pdf,
                    color = cpal[2],
                    xticks = (xticks, tick_labels),
                    margin = 6Plots.mm
                )
            end

            if N2 > 0
                histogram!(filtered2;
                    bins = edges,
                    alpha = 0.5,
                    label = "$lab2. N = $N2",
                    legend = :topright,
                    normalize = :pdf,
                    color = cpal[8]
                )
            end
        end

        mkpath(path)
        savefig("$path/$(safe_title).png")
        if show_fig
            display(current())
        end
    end

    
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
    rowwise_summary. 
    Computes row-wise summary statistics (mean, median, and standard deviation) for a matrix-like data structure, skipping `nothing` values.

    args:
    - `data`: A 2D array (e.g., `Matrix{Union{T, Nothing}}`) where each row represents a set of values. `nothing` values are ignored in the computations.

    returns:
    - `means`: A vector of the mean of each row (ignoring `nothing`).
    - `medians`: A vector of the median of each row (ignoring `nothing`).
    - `stds`: A vector of the standard deviation of each row (ignoring `nothing`).

    notes:
    - Rows with only `nothing` values are skipped entirely and do not contribute to the output.
    - Assumes all rows are of the same length.
    """

    function rowwise_summary(data)
        means   = Float64[]
        medians = Float64[]
        stds    = Float64[]

        for i in 1:size(data, 1)
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
    Included in BooleanNetwork module

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
    mut_robustness_population: a function that computes the mutational robustness
    of a list of matrices

    args:
    - Ws: Population of matrices
    - mr, σr: distribution parameters
    - initial_states: list of vectors corresponding
    - max_steps: maximum number of steps for development
    - activation: activation function
    - trials: number of repetitions of mutation

    returns:
    - robustness_scores: A vector of Float64 for each matrix robustness score
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
    con_robustness_population: does the same thing as mut_robustness_population,
    but with connectivity mutation instead of weight mutation
    """
    function con_robustness_population(Ws::Vector{Matrix{Float64}},
        mr::Float64,
        σr::Float64,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int)

        n = length(Ws)
        rows, cols = size(Ws[1])
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

            sampled_inds = [(rand(1:rows), rand(1:cols)) for _ in 1:trials]
            sampled_vals = rand(d, trials)
            invariant_count = 0

            for t in 1:trials
                i,j = sampled_inds[t]
                old_val = W[i,j]

                if iszero(old_val)
                    W[i,j] = sampled_vals[t]
                else
                    W[i,j] = 0.
                end

                phen, _ = develop(W, state, max_steps, activation)

                if phen !== nothing && phen == orig_phen
                    invariant_count += 1
                end

                W[i,j] = old_val  # Restore original weight
            end

            robustness_scores[i] = invariant_count / trials
        end

        return robustness_scores
    end
    """
    both_robustness_population: does the same thing as mut_robustness_population,
    but with weight and connectivity mutations
    """
    function both_robustness_population(Ws::Vector{Matrix{Float64}},
        mr::Float64,
        σr::Float64,
        initial_states::Matrix{Float64},
        max_steps::Int,
        activation::Function,
        trials::Int)

        n = length(Ws)
        robustness_scores = zeros(Float64, n)
        d = Normal(mr, σr)
        N_genes = length(Ws[1])

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

            sampled_inds_mut = rand(nz_inds, trials)
            sampled_vals_mut = rand(d, trials)

            sampled_inds = rand(N_genes, trials)
            sampled_vals = rand(d, trials)

            invariant_count = 0
            for t in 1:trials
                idx_mut = sampled_inds_mut[t]
                old_val_mut = W[idx_mut]
                W[idx_mut] = sampled_vals_mut[t]

                idx_con = sampled_inds_con[t]
                old_val_con = W[idx_con]
                if old_val == 0.
                    W[idx_con] = sampled_vals_con[t]
                else
                    W[idx_con] = 0.
                end

                phen, _ = develop(W, state, max_steps, activation)

                if phen !== nothing && phen == orig_phen
                    invariant_count += 1
                end

                W[idx_mut] = old_val_mut  # Restore original weight
                W[idx_con] = old_val_con
            end

            robustness_scores[i] = invariant_count / trials
        end

        return robustness_scores
    end

    """
    noise_robustness_population: a function that computes the noise robustness
    of a list of matrices

    args:
    - Ws: Population of matrices
    - noise_dist: distribution from which to sample the perturbations
    - initial_states: list of vectors corresponding to the initial states
    - max_steps: maximum number of steps for development
    - activation: activation function
    - trials: number of repetitions of mutation

    returns:
    - robustness_scores: A vector of Float64 for each matrix robustness score
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

    """
    function compute_density(mat)
        adj = get_nonzero(mat, positive = true)
        return count(adj) / length(adj)
    end

    function get_densities(data)
        gens,pop_size = size(data)

        densities = Matrix{Float64}(undef, gens,pop_size)

        for i in 1:gens
            densities[i,:] .= compute_density.(data[i,:])
        end
        return densities
    end

    """
    ----------------
    SUMMARIZING FUNCTIONS
    -----------------
    """
    function summarize_experiment_data(data)

        processed_data = Dict()

        avg_fit, median_fit, std_fit = rowwise_summary(data["fitness"])
        avg_path, median_path, std_path = rowwise_summary(data["path_length"])

        densities = get_densities(data["matrices"])
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