# Smoothed Moments Estimation

Smoothed Moments estimation, developed by Smith (2005, 2011), provides an alternative approach to HAC estimation that automatically ensures positive semi-definiteness by smoothing moment conditions before taking outer products.

## Mathematical Foundation

### The Smoothed Moments Method

Unlike traditional HAC estimators that compute:

```math
\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^{q} k\left(\frac{j}{S_T}\right) \left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right]
```

Smoothed moments first smooths the moment conditions:

```math
\tilde{g}_t = \sum_{s=-\infty}^{\infty} w_s g_{t-s}
```

Then computes the covariance from smoothed moments:

```math
\hat{\Omega}_{SM} = c \cdot \frac{1}{T} \sum_{t=1}^T \tilde{g}_t \tilde{g}_t'
```

where $w_s$ are kernel weights and $c$ is a normalization constant. The smoothing weights are constructed as:

```math
w_s = \frac{1}{S_T} k\left(\frac{s}{S_T}\right)
```

where $k(\cdot)$ is the kernel function and $S_T$ is the bandwidth.

## Core Types

```@docs
SmoothedMoments
UniformSmoother
TriangularSmoother
```

## Available Kernels

### Uniform Kernel

The uniform kernel provides equal weighting within the bandwidth:

```math
k(x) = \begin{cases}
1 & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**

- Induces Bartlett HAC asymptotically
- Simple computation
- Good for moderate serial correlation

**Usage:**

```julia
using CovarianceMatrices

# Automatic bandwidth (T^(1/3) scaling)
sm_uniform = SmoothedMoments(UniformSmoother())

# Fixed bandwidth
sm_uniform_fixed = SmoothedMoments(UniformSmoother(), 8.0)
```

### Triangular Kernel

The triangular kernel uses linearly decreasing weights:

```math
k(x) = \begin{cases}
1 - |x| & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**

- Induces Parzen HAC asymptotically
- Smoother weighting than uniform
- Better for strong serial correlation

**Usage:**

```julia
# Automatic bandwidth (T^(1/5) scaling)
sm_triangular = SmoothedMoments(TriangularSmoother())

# Fixed bandwidth
sm_triangular_fixed = SmoothedMoments(TriangularSmoother(), 12.0)
```

## Automatic Bandwidth Selection

### Optimal Bandwidth Scaling

The package implements optimal bandwidth scaling for each kernel:

**Uniform Kernel:**

```math
S_T^* = 2.0 \cdot T^{1/3}
```

**Triangular Kernel:**

```math
S_T^* = 1.5 \cdot T^{1/5}
```

These rates are theoretically optimal for the respective kernels.

### Usage Examples

```julia
using CovarianceMatrices, Random
Random.seed!(123)

# Generate AR(1) time series
T = 300
ρ = 0.6
X = zeros(T, 3)
for t in 2:T
    X[t, :] = ρ * X[t-1, :] + randn(3)
end

# Automatic bandwidth selection
sm_auto_uniform = SmoothedMoments(UniformSmoother())
sm_auto_triangular = SmoothedMoments(TriangularSmoother())

Ω_uniform = aVar(sm_auto_uniform, X)
Ω_triangular = aVar(sm_auto_triangular, X)

println("Uniform kernel trace: $(round(tr(Ω_uniform), digits=3))")
println("Triangular kernel trace: $(round(tr(Ω_triangular), digits=3))")

# Check eigenvalues (should all be positive)
println("Uniform min eigenvalue: $(round(minimum(eigvals(Ω_uniform)), digits=6))")
println("Triangular min eigenvalue: $(round(minimum(eigvals(Ω_triangular)), digits=6))")
```

## Threading and Performance

### Automatic Threading

The implementation includes intelligent threading that activates based on sample size:

```julia
# Default: automatic threading for T > 800
sm_default = SmoothedMoments()

# Force single-threaded
sm_single = SmoothedMoments(threaded=false)

# Force multi-threaded
sm_multi = SmoothedMoments(threaded=true)
```

### Performance Characteristics

The kernel-based implementation provides excellent performance:

```julia
using BenchmarkTools

# Performance comparison
T_sizes = [100, 500, 1000, 5000]

for T in T_sizes
    X = randn(T, 4)

    # Smoothed moments
    sm = SmoothedMoments()
    t_sm = @belapsed aVar($sm, $X)

    # Compare with HAC
    hac = Bartlett{Andrews}()
    t_hac = @belapsed aVar($hac, $X)

    println("T=$T: Smoothed=$(round(t_sm*1000, digits=2))ms, HAC=$(round(t_hac*1000, digits=2))ms")
end
```

## Comparison with HAC Estimators

### Theoretical Relationship

Smoothed moments estimators are asymptotically equivalent to their HAC counterparts:

- `SmoothedMoments(UniformSmoother())` ≡ `Bartlett{Andrews}()` (asymptotically)
- `SmoothedMoments(TriangularSmoother())` ≡ `Parzen{Andrews}()` (asymptotically)

### Empirical Comparison

```julia
# Generate strongly autocorrelated data
T = 500
X_persistent = cumsum(randn(T, 3), dims=1)  # Random walk

# Smoothed moments
sm_uniform = SmoothedMoments(UniformSmoother())
sm_triangular = SmoothedMoments(TriangularSmoother())

# Corresponding HAC estimators
hac_bartlett = Bartlett{Andrews}()
hac_parzen = Parzen{Andrews}()

# Compare results
estimators = [
    ("Smoothed Uniform", sm_uniform),
    ("HAC Bartlett", hac_bartlett),
    ("Smoothed Triangular", sm_triangular),
    ("HAC Parzen", hac_parzen)
]

println("Estimator\t\tTrace\tMin Eigenvalue\tCondition Number")
for (name, est) in estimators
    Ω = aVar(est, X_persistent)
    eigenvals = eigvals(Ω)

    trace_val = tr(Ω)
    min_eig = minimum(eigenvals)
    cond_num = maximum(eigenvals) / min_eig

    println("$(rpad(name, 20))\t$(round(trace_val, digits=2))\t$(round(min_eig, digits=5))\t\t$(round(cond_num, digits=1))")
end
```

## Advanced Usage

### Custom Bandwidth Selection

```julia
# Bandwidth sensitivity analysis
bandwidths = [5.0, 10.0, 15.0, 20.0]

println("Bandwidth\tTrace\tCondition Number")
for bw in bandwidths
    sm = SmoothedMoments(UniformSmoother(), bw)
    Ω = aVar(sm, X)
    println("$bw\t\t$(round(tr(Ω), digits=3))\t$(round(cond(Ω), digits=1))")
end
```

### Memory Efficiency Analysis

The kernel-based approach is significantly more memory efficient:

```julia
# Memory usage comparison
function memory_comparison(T)
    X = randn(T, 4)

    # Traditional weight-based approach would need 2T-1 weights
    traditional_memory = (2*T - 1) * sizeof(Float64)

    # Kernel-based approach needs minimal memory
    kernel_memory = T * 4 * sizeof(Float64)  # Just for temporary column

    reduction = (1 - kernel_memory / traditional_memory) * 100

    println("T=$T: Traditional=$(traditional_memory÷1024)KB, Kernel=$(kernel_memory÷1024)KB")
    println("Memory reduction: $(round(reduction, digits=1))%")
end

for T in [500, 1000, 5000, 10000]
    memory_comparison(T)
    println()
end
```

## Integration with GLM

Smoothed moments work seamlessly with GLM regression models:

```julia
using GLM, DataFrames

# Generate time series regression data
T = 200
x1 = cumsum(randn(T)) / sqrt(T)  # Random walk regressor
x2 = sin.(2π * (1:T) / 12) + randn(T) * 0.1  # Seasonal pattern
ε = cumsum(randn(T)) * 0.3  # AR errors

y = 2.0 .+ 1.5 * x1 .- 0.8 * x2 .+ ε

df = DataFrame(y=y, x1=x1, x2=x2)
model = lm(@formula(y ~ x1 + x2), df)

# Compare standard errors
se_classical = stderror(model)
se_smoothed_uniform = stderror(SmoothedMoments(UniformSmoother()), model)
se_smoothed_triangular = stderror(SmoothedMoments(TriangularSmoother()), model)
se_hac_bartlett = stderror(Bartlett{Andrews}(), model)

results = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Classical = round.(se_classical, digits=4),
    Smoothed_Uniform = round.(se_smoothed_uniform, digits=4),
    Smoothed_Triangular = round.(se_smoothed_triangular, digits=4),
    HAC_Bartlett = round.(se_hac_bartlett, digits=4)
)

println(results)

# Check closeness of smoothed and HAC
println("\nRatio Smoothed/HAC (should be close to 1):")
println("Uniform/Bartlett: ", round.(se_smoothed_uniform ./ se_hac_bartlett, digits=3))
```

## Practical Guidelines

### When to Use Smoothed Moments

✅ **Recommended for:**

- Applications requiring guaranteed positive semi-definiteness
- Time series with moderate to strong serial correlation
- When computational efficiency is important
- Exploratory analysis where robustness is key
- Applications sensitive to numerical stability

✅ **Particularly good for:**

- Financial time series
- Macro-econometric models
- GMM estimation
- Bootstrap procedures (maintains PSD across samples)

### Kernel Choice Guidelines

```julia
# Decision framework
function choose_kernel(X, criterion="trace_stability")
    T, k = size(X)

    sm_uniform = SmoothedMoments(UniformSmoother())
    sm_triangular = SmoothedMoments(TriangularSmoother())

    Ω_uniform = aVar(sm_uniform, X)
    Ω_triangular = aVar(sm_triangular, X)

    if criterion == "trace_stability"
        # Choose based on condition number
        cond_uniform = cond(Ω_uniform)
        cond_triangular = cond(Ω_triangular)

        if cond_uniform < cond_triangular
            return "UniformSmoother", Ω_uniform
        else
            return "TriangularSmoother", Ω_triangular
        end

    elseif criterion == "eigenvalue_spread"
        # Choose based on eigenvalue spread
        eig_uniform = eigvals(Ω_uniform)
        eig_triangular = eigvals(Ω_triangular)

        spread_uniform = maximum(eig_uniform) / minimum(eig_uniform)
        spread_triangular = maximum(eig_triangular) / minimum(eig_triangular)

        if spread_uniform < spread_triangular
            return "UniformSmoother", Ω_uniform
        else
            return "TriangularSmoother", Ω_triangular
        end
    end
end

# Example usage
choice, Ω_chosen = choose_kernel(X, "trace_stability")
println("Recommended kernel: $choice")
```

### Sample Size Considerations

```julia
# Sample size guidelines
function sample_size_recommendations()
    recommendations = [
        (50, "Consider using fixed bandwidth"),
        (100, "Automatic bandwidth usually works well"),
        (300, "Both kernels typically perform well"),
        (1000, "Threading becomes beneficial"),
        (5000, "Excellent performance with either kernel")
    ]

    println("Sample Size Guidelines:")
    for (T, rec) in recommendations
        println("T ≥ $T: $rec")
    end
end

sample_size_recommendations()
```

### Troubleshooting

**Issue: Results differ significantly from HAC**

```julia
# Check if bandwidth scaling is appropriate
function bandwidth_diagnostic(X)
    T, k = size(X)

    sm_uniform = SmoothedMoments(UniformSmoother())
    Ω_sm = aVar(sm_uniform, X)

    hac_bartlett = Bartlett{Andrews}()
    Ω_hac = aVar(hac_bartlett, X)

    # Get optimal bandwidth for HAC
    _, _, bw_hac = workingoptimalbw(hac_bartlett, X)

    # Compare with automatic smoothed moments bandwidth
    bw_sm_auto = 2.0 * T^(1/3)

    println("HAC bandwidth: $(round(bw_hac, digits=2))")
    println("SM auto bandwidth: $(round(bw_sm_auto, digits=2))")
    println("Ratio: $(round(bw_sm_auto / bw_hac, digits=2))")

    # Try matching bandwidth
    sm_matched = SmoothedMoments(UniformSmoother(), bw_hac)
    Ω_matched = aVar(sm_matched, X)

    println("\nTrace comparison:")
    println("SM (auto): $(round(tr(Ω_sm), digits=3))")
    println("HAC: $(round(tr(Ω_hac), digits=3))")
    println("SM (matched): $(round(tr(Ω_matched), digits=3))")
end

bandwidth_diagnostic(X)
```

## Implementation Details

### Kernel-Based Computation

The implementation uses efficient kernel-based computation:

```julia
# Example of the kernel-based approach (conceptual)
function smooth_moments_conceptual(G, kernel::UniformSmoother, bandwidth, T)
    T_data, m = size(G)
    max_lag = floor(Int, bandwidth)

    result = similar(G)
    for j in 1:m
        for t in 1:T_data
            smooth_val = 0.0
            # Sum over bandwidth window with uniform weights
            for lag in -max_lag:max_lag
                source_idx = t - lag
                if 1 ≤ source_idx ≤ T_data
                    smooth_val += G[source_idx, j]
                end
            end
            result[t, j] = smooth_val / bandwidth
        end
    end
    return result
end

# The actual implementation is optimized with threading and memory efficiency
```

### Memory Optimization

Key optimizations include:

1. **No weight storage**: Weights computed on-the-fly
2. **Temporary buffers**: Minimal memory usage
3. **Column-wise processing**: Cache-friendly access patterns
4. **In-place operations**: Reduced allocations

## References

**Primary References:**

- Smith, R.J. (2005). "Automatic positive semidefinite HAC covariance matrix and GMM estimation". _Econometric Theory_, 21(1), 158-170.
- Smith, R.J. (2011). "GEL criteria for moment condition models". _Econometric Theory_, 27(6), 1192-1235.

**Related Work:**

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". _Econometrica_, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". _Econometrica_, 59(3), 817-858.
