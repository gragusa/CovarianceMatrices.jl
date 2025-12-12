# Clustered Robust (CR) Estimators

Clustered robust estimators provide consistent covariance estimates when observations are correlated within clusters but independent across clusters. This is particularly important in panel data, multi-level data, and situations with natural grouping structures.

## Mathematical Foundation

### The Clustering Problem

When observations are correlated within clusters, the standard independence assumption breaks down. For data clustered by groups $g \in \{1, \ldots, G\}$, the long-run covariance matrix becomes:

```math
\Omega = \sum_{g=1}^G \mathbb{E}\left[\left(\sum_{t \in g} g(z_t, \theta_0)\right)\left(\sum_{t \in g} g(z_t, \theta_0)\right)'\right]
```

This captures the fact that within-cluster correlations can be arbitrary, while between-cluster correlations are assumed to be zero.

### CR Estimator Family

The general form of CR estimators is:

```math
\hat{\Omega}_{CR_j} = \frac{G}{G-1} \sum_{g=1}^G \phi_j(g) \hat{u}_g \hat{u}_g'
```

where:
- $\hat{u}_g = \sum_{t \in g} g_t$ is the cluster-level sum of moment conditions
- $\phi_j(g)$ are small-sample adjustment factors
- $\frac{G}{G-1}$ is a degrees-of-freedom correction

## Available Estimators

```@docs
CR0
CR1
CR2
CR3
```

### CR0: Basic Cluster Estimator

```math
\phi_0(g) = 1
```

**Properties:**
- No small-sample adjustments
- Simplest form of cluster-robust inference
- Can be downward biased in small samples

**Usage:**
```julia
using CovarianceMatrices

# Single cluster variable
cluster_ids = [1, 1, 2, 2, 3, 3, 4, 4]
cr0 = CR0(cluster_ids)
Ω = aVar(cr0, residuals)
```

### CR1: Degrees-of-Freedom Correction

```math
\phi_1(g) = \frac{G}{G-1} \cdot \frac{N-1}{N-K}
```

**Properties:**
- Applies degrees-of-freedom correction for parameters
- Better finite sample properties than CR0
- **Most commonly used in practice**

**Usage:**
```julia
# CR1 with single clustering dimension
cr1 = CR1(cluster_ids)
Ω = aVar(cr1, residuals)
```

### CR2 and CR3: Leverage-Based Corrections

**CR2:**
```math
\phi_2(g) = \frac{1}{1-h_g}
```

**CR3:**
```math
\phi_3(g) = \frac{1}{(1-h_g)^2}
```

where $h_g$ represents cluster-level leverage values.

**Properties:**
- Account for cluster-specific leverage effects
- More robust to influential clusters
- Better performance with unbalanced clusters

## Multi-Way Clustering

The package supports multi-way clustering with intersection corrections:

```julia
# Two-way clustering
firm_ids = [1, 1, 2, 2, 3, 3, 4, 4]
year_ids = [2001, 2002, 2001, 2002, 2001, 2002, 2001, 2002]

# Multi-way clustering
cr_multi = CR1((firm_ids, year_ids))
Ω = aVar(cr_multi, residuals)
```

The estimator automatically applies the inclusion-exclusion principle:
```math
\hat{\Omega}_{multi} = \hat{\Omega}_{firm} + \hat{\Omega}_{year} - \hat{\Omega}_{firm \cap year}
```

## Practical Usage Examples

### Panel Data Example

```julia
using CovarianceMatrices, DataFrames, GLM, Random
Random.seed!(123)

# Generate panel data
N_firms = 50
N_years = 10
N = N_firms * N_years

# Create panel structure
firm_ids = repeat(1:N_firms, N_years)
year_ids = repeat(1:N_years, inner=N_firms)

# Generate data with firm and time effects
firm_effects = randn(N_firms)[firm_ids]
time_effects = randn(N_years)[year_ids]
X = randn(N, 2)
ε = firm_effects + time_effects + randn(N) * 0.5

y = 1.0 .+ 0.5 * X[:, 1] - 0.3 * X[:, 2] + ε

df = DataFrame(
    y = y,
    x1 = X[:, 1],
    x2 = X[:, 2],
    firm_id = firm_ids,
    year_id = year_ids
)

# Fit model
model = lm(@formula(y ~ x1 + x2), df)

# Compare different clustering approaches
se_classical = stderror(model)
se_firm = stderror(CR1(df.firm_id), model)
se_year = stderror(CR1(df.year_id), model)
se_twoway = stderror(CR1((df.firm_id, df.year_id)), model)

results = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Classical = round.(se_classical, digits=4),
    Firm_Cluster = round.(se_firm, digits=4),
    Year_Cluster = round.(se_year, digits=4),
    TwoWay_Cluster = round.(se_twoway, digits=4)
)

println(results)
```

### Comparing CR Variants

```julia
# Compare all CR estimators
cr_estimators = [CR0(firm_ids), CR1(firm_ids), CR2(firm_ids), CR3(firm_ids)]
cr_names = ["CR0", "CR1", "CR2", "CR3"]

println("Estimator\\tStd Errors (x1, x2)")
for (est, name) in zip(cr_estimators, cr_names)
    se = stderror(est, model)
    println("$name\\t\\t$(round.(se[2:3], digits=4))")
end
```

### Cluster Diagnostics

```julia
function cluster_diagnostics(cluster_var)
    n_clusters = length(unique(cluster_var))
    cluster_sizes = [sum(cluster_var .== g) for g in unique(cluster_var)]

    println("Cluster Diagnostics:")
    println("Number of clusters: $n_clusters")
    println("Average cluster size: $(round(mean(cluster_sizes), digits=1))")
    println("Min cluster size: $(minimum(cluster_sizes))")
    println("Max cluster size: $(maximum(cluster_sizes))")

    # Rule of thumb: need at least 30-50 clusters for asymptotic theory
    if n_clusters < 30
        println("⚠️  Warning: Few clusters detected. Consider bootstrap inference.")
    end

    return n_clusters, cluster_sizes
end

n_clusters, sizes = cluster_diagnostics(firm_ids)
```

## Selection Guidelines

### When to Use Clustered Standard Errors

✅ **Use CR estimators for:**
- Panel data with repeated observations per unit
- Multi-level data (students within schools, firms within industries)
- Spatial data with geographic clustering
- Survey data with complex sampling designs
- Any situation with natural grouping structures

### CR Variant Selection

```julia
function choose_cr_estimator(n_clusters, cluster_balance="balanced")
    if n_clusters < 30
        println("Recommendation: Use CR1 with bootstrap inference")
        return "CR1 + Bootstrap"
    elseif n_clusters < 50
        println("Recommendation: Use CR1 (conservative)")
        return "CR1"
    elseif cluster_balance == "unbalanced"
        println("Recommendation: Use CR2 or CR3 (leverage corrections)")
        return "CR2"
    else
        println("Recommendation: CR1 is adequate")
        return "CR1"
    end
end

# Apply to our example
recommendation = choose_cr_estimator(length(unique(firm_ids)))
```

### Multi-Way Clustering Decision

```julia
# Test if two-way clustering is necessary
function test_clustering_necessity(model, cluster1, cluster2)
    # Single-way clustering
    se1 = stderror(CR1(cluster1), model)
    se2 = stderror(CR1(cluster2), model)

    # Two-way clustering
    se_2way = stderror(CR1((cluster1, cluster2)), model)

    # Compare max single-way vs two-way
    se_max_single = max.(se1, se2)

    println("Two-way vs Max Single-way Comparison:")
    for i in 1:length(se1)
        ratio = se_2way[i] / se_max_single[i]
        println("Coef $i: $(round(ratio, digits=3))")
        if ratio > 1.1
            println("  → Two-way clustering provides additional correction")
        end
    end
end

test_clustering_necessity(model, df.firm_id, df.year_id)
```

## Advanced Features

### Unbalanced Clustering

```julia
# Handle unbalanced clusters automatically
unbalanced_clusters = vcat(fill(1, 100), fill(2, 10), fill(3, 5))
cr_unbalanced = CR2(unbalanced_clusters)  # Use leverage corrections
```

### Nested Clustering

```julia
# For hierarchical structures (students in classes in schools)
school_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3]
class_ids = [1, 1, 2, 3, 3, 4, 5, 5, 6]

# Cluster at the highest level (schools)
cr_nested = CR1(school_ids)
```

### Performance Considerations

```julia
using BenchmarkTools

function cr_performance_comparison()
    N = 10000
    clusters = repeat(1:100, inner=100)
    residuals_perf = randn(N, 3)

    println("Performance Comparison:")

    for (name, estimator) in [("CR0", CR0(clusters)), ("CR1", CR1(clusters))]
        time = @belapsed aVar($estimator, $residuals_perf)
        println("$name: $(round(time * 1000, digits=2))ms")
    end
end

cr_performance_comparison()
```

## CachedCR: High-Performance Caching for Repeated Calculations

For applications requiring repeated cluster-robust variance calculations with the same cluster structure—such as wild bootstrap or Monte Carlo simulations—`CachedCR` provides significant performance improvements.

```@docs
CachedCR
CRCache
```

### When to Use CachedCR

`CachedCR` is designed for scenarios where:
- The same cluster structure is used repeatedly
- Only the moment matrix (residuals) changes between calculations
- Performance is critical (e.g., bootstrap with 1000+ replications)

**Typical speedups:**
- Single-cluster: ~3-4x faster
- Two-way clustering: ~4x faster
- Wild bootstrap (100 iterations): ~2.5-4x faster

### Basic Usage

```julia
using CovarianceMatrices

# Setup
cluster_ids = repeat(1:100, inner=10)
X = randn(1000, 5)

# Create standard CR estimator
k = CR0(cluster_ids)

# Create cached version (one-time setup cost)
cached_k = CachedCR(k, size(X, 2))

# Use for repeated calculations
S = aVar(cached_k, X)  # Uses optimized gather-based aggregation
```

### Wild Bootstrap Example

```julia
using CovarianceMatrices, Random

function wild_bootstrap_variance(X, cluster_ids; n_bootstrap=1000)
    n = size(X, 1)
    ncols = size(X, 2)

    # Create cached estimator (one-time cost)
    k = CR0(cluster_ids)
    cached_k = CachedCR(k, ncols)

    # Store bootstrap variance estimates
    bootstrap_vars = Vector{Matrix{Float64}}(undef, n_bootstrap)

    for b in 1:n_bootstrap
        # Rademacher weights (+1 or -1)
        weights = rand([-1.0, 1.0], n)
        X_perturbed = X .* weights

        # Fast variance calculation using cache
        bootstrap_vars[b] = aVar(cached_k, X_perturbed)
    end

    return bootstrap_vars
end

# Usage
cluster_ids = repeat(1:50, inner=20)
X = randn(1000, 5)
vars = wild_bootstrap_variance(X, cluster_ids; n_bootstrap=1000)
```

### Two-Way Clustering with Cache

```julia
# Panel data: firms × years
n_firms, n_years = 100, 20
n_obs = n_firms * n_years

firm_ids = repeat(1:n_firms, outer=n_years)
year_ids = repeat(1:n_years, inner=n_firms)

X = randn(n_obs, 4)

# Create two-way clustered estimator with cache
k = CR0((firm_ids, year_ids))
cached_k = CachedCR(k, size(X, 2))

# The cache precomputes GroupedArrays for all combinations:
# - firm clustering
# - year clustering
# - firm × year intersection
S = aVar(cached_k, X)
```

### Important Limitations

!!! warning "AD Incompatibility"
    `CachedCR` uses in-place operations and preallocated buffers that are **not compatible with automatic differentiation (AD)**. For AD-compatible code, use the standard `CR0`, `CR1`, `CR2`, or `CR3` estimators directly.

!!! note "Fixed Column Count"
    The cache is built for a specific number of columns. Using a moment matrix with a different column count will raise an error:
    ```julia
    cached_k = CachedCR(CR0(clusters), 5)  # Cache for 5 columns
    aVar(cached_k, randn(100, 3))  # Error! Expected 5 columns
    ```

### Performance Benchmark

```julia
using BenchmarkTools

function benchmark_cached_cr()
    n_obs = 10_000
    n_cols = 10
    n_clusters = 100

    cluster_ids = repeat(1:n_clusters, inner=n_obs ÷ n_clusters)
    X = randn(n_obs, n_cols)

    k = CR0(cluster_ids)
    cached_k = CachedCR(k, n_cols)

    println("Standard CR0:")
    @btime aVar($k, $X)

    println("CachedCR:")
    @btime aVar($cached_k, $X)
end

benchmark_cached_cr()
# Typical output:
#   Standard CR0:  420.000 μs (5 allocations: 9.00 KiB)
#   CachedCR:      105.000 μs (1 allocation: 896 bytes)
```

## Integration with Other Estimators

### Combining with HAC

For panel data with both clustering and serial correlation:

```julia
# First handle clustering, then serial correlation within clusters
# (This requires specialized estimators not covered by basic CR)
println("Note: For panel data with both clustering and time dependence,")
println("consider specialized panel-robust estimators like Driscoll-Kraay.")
```

### Bootstrap Inference

```julia
function bootstrap_cluster_inference(model, clusters, n_bootstrap=1000)
    # Simple cluster bootstrap (resample clusters)
    unique_clusters = unique(clusters)
    n_clusters = length(unique_clusters)

    bootstrap_coefs = []

    for b in 1:n_bootstrap
        # Resample clusters with replacement
        boot_clusters = sample(unique_clusters, n_clusters, replace=true)

        # Create bootstrap sample
        boot_indices = Int[]
        for cluster in boot_clusters
            append!(boot_indices, findall(clusters .== cluster))
        end

        # Re-estimate (simplified - would need actual re-estimation)
        # This is a conceptual example
        push!(bootstrap_coefs, randn(3))  # Placeholder
    end

    println("Bootstrap inference recommended for < 30 clusters")
    return bootstrap_coefs
end
```

## Best Practices Summary

### Default Recommendations

1. **Standard choice**: Use `CR1()` for most applications
2. **Few clusters (< 30)**: Consider bootstrap inference
3. **Unbalanced clusters**: Use `CR2()` or `CR3()`
4. **Multi-way clustering**: Test necessity before applying
5. **Panel data**: Consider Driscoll-Kraay for serial correlation

### Common Pitfalls

❌ **Avoid:**
- Using CR with too few clusters (< 30)
- Ignoring cluster balance issues
- Over-clustering (more cluster dimensions than necessary)
- Using CR when independence assumption holds

✅ **Do:**
- Examine cluster structure and balance
- Test clustering necessity
- Consider bootstrap for small cluster samples
- Document clustering rationale

## References

**Foundational Papers:**
- Liang, K.Y. and Zeger, S.L. (1986). "Longitudinal data analysis using generalized linear models". *Biometrika*, 73(1), 13-22.
- White, H. (1984). "Asymptotic Theory for Econometricians". Academic Press.

**Methodological Extensions:**
- Cameron, A.C., Gelbach, J.B., and Miller, D.L. (2011). "Robust inference with multiway clustering". *Journal of Business & Economic Statistics*, 29(2), 238-249.
- Petersen, M.A. (2009). "Estimating standard errors in finance panel data sets: Comparing approaches". *Review of Financial Studies*, 22(1), 435-480.

**Small Sample Properties:**
- Cameron, A.C. and Miller, D.L. (2015). "A practitioner's guide to cluster-robust inference". *Journal of Human Resources*, 50(2), 317-372.