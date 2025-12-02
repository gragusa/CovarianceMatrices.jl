# Benchmark: CachedCR vs standard CR implementations
#
# This benchmark compares the performance of the cached cluster-robust
# variance estimator (CachedCR) against the standard implementation.
#
# Run with: julia --project benchmark/benchmark_cached_cr.jl

using CovarianceMatrices
using BenchmarkTools
using Random
using Printf

# Disable interpolation warnings for cleaner output
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0

# Helper function for bootstrap benchmark (must be at module level for @benchmark)
function run_bootstrap!(results, k, X, weights_list)
    for (i, w) in enumerate(weights_list)
        results[i] = aVar(k, X .* w)
    end
    return results
end

function print_header(title::String)
    println("\n", "="^70)
    println(title)
    println("="^70)
end

function print_comparison(name::String, t_standard, t_cached)
    speedup = median(t_standard).time / median(t_cached).time
    alloc_reduction = median(t_standard).allocs / max(median(t_cached).allocs, 1)

    println("\n$name:")
    println("  Standard:  $(median(t_standard))")
    println("  Cached:    $(median(t_cached))")
    @printf("  Speedup:   %.2fx faster\n", speedup)
    @printf("  Allocs:    %.1fx fewer allocations\n", alloc_reduction)
end

function benchmark_single_cluster()
    print_header("Single Cluster Variable")

    # Setup
    n_obs = 10_000
    n_cols = 10
    n_clusters = 100

    Random.seed!(42)
    cluster_ids = repeat(1:n_clusters, inner=n_obs ÷ n_clusters)
    X = randn(n_obs, n_cols)

    k = CR0(cluster_ids)
    cached_k = CachedCR(k, n_cols)

    println("Configuration:")
    println("  Observations: $n_obs")
    println("  Columns: $n_cols")
    println("  Clusters: $n_clusters")

    # Warmup
    aVar(k, X)
    aVar(cached_k, X)

    # Benchmark single call
    t_standard = @benchmark aVar($k, $X)
    t_cached = @benchmark aVar($cached_k, $X)

    print_comparison("Single aVar call", t_standard, t_cached)

    # Benchmark wild bootstrap scenario (many calls, same structure)
    n_bootstrap = 100
    weights_list = [rand([-1.0, 1.0], n_obs) for _ in 1:n_bootstrap]
    results_std = Vector{Matrix{Float64}}(undef, n_bootstrap)
    results_cch = Vector{Matrix{Float64}}(undef, n_bootstrap)

    t_standard_boot = @benchmark run_bootstrap!($results_std, $k, $X, $weights_list)
    t_cached_boot = @benchmark run_bootstrap!($results_cch, $cached_k, $X, $weights_list)

    print_comparison("Wild bootstrap ($n_bootstrap iterations)", t_standard_boot, t_cached_boot)
end

function benchmark_twoway_cluster()
    print_header("Two-Way Clustering (Firm × Year)")

    # Setup: Panel data structure
    n_firms = 100
    n_years = 20
    n_obs = n_firms * n_years
    n_cols = 8

    Random.seed!(123)
    firm_ids = repeat(1:n_firms, outer=n_years)
    year_ids = repeat(1:n_years, inner=n_firms)
    X = randn(n_obs, n_cols)

    k = CR0((firm_ids, year_ids))
    cached_k = CachedCR(k, n_cols)

    println("Configuration:")
    println("  Firms: $n_firms")
    println("  Years: $n_years")
    println("  Observations: $n_obs")
    println("  Columns: $n_cols")

    # Warmup
    aVar(k, X)
    aVar(cached_k, X)

    # Benchmark single call
    t_standard = @benchmark aVar($k, $X)
    t_cached = @benchmark aVar($cached_k, $X)

    print_comparison("Single aVar call", t_standard, t_cached)

    # Benchmark wild bootstrap scenario
    n_bootstrap = 100
    weights_list = [rand([-1.0, 1.0], n_obs) for _ in 1:n_bootstrap]
    results_std = Vector{Matrix{Float64}}(undef, n_bootstrap)
    results_cch = Vector{Matrix{Float64}}(undef, n_bootstrap)

    t_standard_boot = @benchmark run_bootstrap!($results_std, $k, $X, $weights_list)
    t_cached_boot = @benchmark run_bootstrap!($results_cch, $cached_k, $X, $weights_list)

    print_comparison("Wild bootstrap ($n_bootstrap iterations)", t_standard_boot, t_cached_boot)
end

function benchmark_varying_sizes()
    print_header("Scaling with Problem Size (Single Cluster)")

    println("\n┌─────────────┬─────────────┬─────────────┬──────────┐")
    println("│ Observations│   Standard  │    Cached   │  Speedup │")
    println("├─────────────┼─────────────┼─────────────┼──────────┤")

    for n_obs in [1_000, 5_000, 10_000, 50_000]
        n_cols = 5
        n_clusters = n_obs ÷ 10

        Random.seed!(42)
        cluster_ids = repeat(1:n_clusters, inner=10)
        X = randn(n_obs, n_cols)

        k = CR0(cluster_ids)
        cached_k = CachedCR(k, n_cols)

        # Warmup
        aVar(k, X)
        aVar(cached_k, X)

        t_standard = @benchmark aVar($k, $X) samples=50
        t_cached = @benchmark aVar($cached_k, $X) samples=50

        speedup = median(t_standard).time / median(t_cached).time

        t_std_str = @sprintf("%10.2f μs", median(t_standard).time / 1e3)
        t_cch_str = @sprintf("%10.2f μs", median(t_cached).time / 1e3)
        spd_str = @sprintf("%7.2fx", speedup)

        @printf("│ %11d │%s │%s │%s │\n", n_obs, t_std_str, t_cch_str, spd_str)
    end

    println("└─────────────┴─────────────┴─────────────┴──────────┘")
end

function benchmark_cr_variants()
    print_header("CR Variants (CR0, CR1, CR2, CR3)")

    n_obs = 5_000
    n_cols = 5
    n_clusters = 50

    Random.seed!(42)
    cluster_ids = repeat(1:n_clusters, inner=n_obs ÷ n_clusters)
    X = randn(n_obs, n_cols)

    println("Configuration:")
    println("  Observations: $n_obs")
    println("  Columns: $n_cols")
    println("  Clusters: $n_clusters")

    for (name, CR) in [("CR0", CR0), ("CR1", CR1), ("CR2", CR2), ("CR3", CR3)]
        k = CR(cluster_ids)
        cached_k = CachedCR(k, n_cols)

        # Warmup
        aVar(k, X)
        aVar(cached_k, X)

        t_standard = @benchmark aVar($k, $X) samples=50
        t_cached = @benchmark aVar($cached_k, $X) samples=50

        print_comparison(name, t_standard, t_cached)
    end
end

function main()
    println("\n" * "▓"^70)
    println("  CachedCR Benchmark Suite")
    println("  Comparing cached vs standard cluster-robust variance estimation")
    println("▓"^70)

    benchmark_single_cluster()
    benchmark_twoway_cluster()
    benchmark_varying_sizes()
    benchmark_cr_variants()

end

# Run benchmarks
main()
