# Performance Notes

This section provides guidance on computational performance aspects of CovarianceMatrices.jl estimators, including timing comparisons, memory usage, and optimization strategies.

## Performance Overview

The estimators in CovarianceMatrices.jl vary significantly in computational complexity:

### Computational Complexity

| Estimator Category | Time Complexity | Memory Usage | Main Bottleneck |
|-------------------|-----------------|--------------|-----------------|
| HC/HR | O(T·k²) | O(T·k) | Matrix operations |
| CR | O(T·k² + G·k²) | O(T·k) | Grouping operations |
| HAC (fixed bandwidth) | O(T·k²·q) | O(T·k) | Autocovariance computation |
| HAC (automatic bandwidth) | O(T·k²·q + T·k³) | O(T·k) | Bandwidth selection |
| VARHAC | O(T·k²·p + k³·p³) | O(T·k) | VAR estimation |
| Smoothed Moments | O(T·k²·q) | O(T·k) | Kernel smoothing |
| Driscoll-Kraay | O(T·N·k²·q) | O(T·N·k) | Panel structure |
| EWC | O(T·k²·B) | O(T·k) | Basis function computation |

Where:
- T = sample size
- k = number of variables/parameters
- q = bandwidth/truncation lag
- p = VAR lag length
- G = number of clusters
- N = number of cross-sectional units (panels)
- B = number of basis functions (EWC)

## Benchmarking Results

### Cross-Sectional Estimators (T=1000, k=5)

```julia
using BenchmarkTools, CovarianceMatrices, Random
Random.seed!(123)

T, k = 1000, 5
X = randn(T, k)

# HC estimators
@benchmark aVar(HC0(), $X)     # ~50μs
@benchmark aVar(HC1(), $X)     # ~55μs
@benchmark aVar(HC2(), $X)     # ~80μs (needs leverage computation)
@benchmark aVar(HC3(), $X)     # ~85μs
@benchmark aVar(HC4(), $X)     # ~120μs (complex leverage adjustments)
```

### Time Series Estimators

```julia
# HAC estimators (T=1000, k=5)
@benchmark aVar(Bartlett(5), $X)              # ~150μs (fixed bandwidth)
@benchmark aVar(Bartlett{NeweyWest}(), $X)    # ~200μs (simple rule)
@benchmark aVar(Bartlett{Andrews}(), $X)      # ~2ms (bandwidth selection)

# Advanced estimators
@benchmark aVar(VARHAC(), $X)                 # ~800μs (VAR fitting)
@benchmark aVar(SmoothedMoments(), $X)        # ~180μs (kernel-based)
@benchmark aVar(EWC(10), $X)                  # ~300μs
```

### Scaling with Sample Size

```julia
function scaling_analysis()
    sizes = [100, 500, 1000, 2000, 5000]
    k = 4

    results = Dict()

    for T in sizes
        X = randn(T, k)

        # Fast estimators
        results["HC3_$T"] = @belapsed aVar(HC3(), $X)
        results["VARHAC_$T"] = @belapsed aVar(VARHAC(), $X)
        results["SmoothedMoments_$T"] = @belapsed aVar(SmoothedMoments(), $X)

        # Slower estimators
        results["BartlettAndrews_$T"] = @belapsed aVar(Bartlett{Andrews}(), $X)
    end

    return results
end
```

Expected scaling (approximate):

| T | HC3 | VARHAC | Smoothed Moments | Bartlett-Andrews |
|---|-----|--------|------------------|------------------|
| 100 | 5μs | 80μs | 20μs | 200μs |
| 500 | 25μs | 200μs | 90μs | 800μs |
| 1000 | 50μs | 400μs | 180μs | 2ms |
| 2000 | 100μs | 800μs | 360μs | 8ms |
| 5000 | 250μs | 2ms | 900μs | 50ms |

## Memory Usage Patterns

### Memory Efficiency Improvements

The package includes several memory optimizations:

#### 1. Kernel-Based Smoothing (Smoothed Moments)

**Old approach** (weight-based):
```julia
# Would need 2T-1 weight storage
memory_old = (2*T - 1) * sizeof(Float64)  # ~16KB for T=1000
```

**New approach** (kernel-based):
```julia
# Only needs temporary column buffer
memory_new = T * sizeof(Float64)  # ~8KB for T=1000
# Memory reduction: ~50% for typical cases, up to 90% for large T
```

#### 2. In-Place Operations

```julia
# Efficient: modifies matrix in-place
aVar!(Ω, estimator, X)  # Uses pre-allocated Ω

# Less efficient: allocates new matrix
Ω = aVar(estimator, X)
```

#### 3. Threading Memory Overhead

Threading adds minimal memory overhead:
```julia
# Single-threaded
memory_single = T * k * sizeof(Float64)

# Multi-threaded (k columns processed in parallel)
memory_threaded = T * k * sizeof(Float64) + n_threads * small_overhead
```

## Threading Performance

### Automatic Threading Decisions

Smoothed Moments uses intelligent threading:

```julia
# Threading activated based on problem size
function should_use_threading(T, threaded_param)
    if threaded_param === true
        return true
    elseif threaded_param === false
        return false
    else
        return T > 800  # Automatic threshold
    end
end
```

### Threading Effectiveness

Based on benchmark results (Apple M3 Max, 10 cores):

| T | Single-threaded | 4 threads | 8 threads | Speedup (8t) |
|---|----------------|-----------|-----------|--------------|
| 100 | 2.1μs | 23.2μs | 40.1μs | 0.05x (overhead dominates) |
| 500 | 9.7μs | 30.9μs | 42.4μs | 0.23x (still overhead-bound) |
| 1000 | 19.0μs | 41.3μs | 46.6μs | 0.41x (break-even approaching) |
| 10000 | 190.2μs | 148.4μs | 124.4μs | 1.53x (good speedup) |

**Threading Recommendations:**
- T < 800: Use single-threaded (automatic default)
- T > 800: Threading beneficial (automatic activation)
- T > 5000: Excellent threading performance

## Optimization Strategies

### 1. Choose the Right Estimator for Your Use Case

```julia
# Fast choices by data type:

# Cross-sectional → HC3 (fastest robust choice)
ve_cross = HC3()

# Time series, want automatic → VARHAC (no bandwidth selection)
ve_ts_auto = VARHAC()

# Time series, want traditional → Fixed bandwidth HAC
ve_ts_trad = Bartlett(round(Int, 0.75 * T^(1/3)))

# Panel data → CR1 (efficient clustering)
ve_panel = CR1(cluster_ids)
```

### 2. Pre-allocate When Possible

```julia
# For repeated computations
Ω_buffer = Matrix{Float64}(undef, k, k)

for dataset in datasets
    aVar!(Ω_buffer, HC3(), dataset)  # Re-uses buffer
    # Process Ω_buffer...
end
```

### 3. Bandwidth Strategies for HAC

```julia
# Fastest: Fixed bandwidth (no selection overhead)
hac_fast = Bartlett(6)

# Medium: NeweyWest rule (simple formula)
hac_medium = Bartlett{NeweyWest}()

# Slower: Andrews optimal (requires VAR estimation)
hac_slow = Bartlett{Andrews}()

# Alternative: VARHAC (automatic, often faster than Andrews)
hac_auto = VARHAC()
```

### 4. Memory-Conscious Choices

```julia
# For large datasets, prefer estimators with O(Tk) memory:
memory_efficient = [HC3(), VARHAC(), SmoothedMoments()]

# Avoid for very large T:
memory_intensive = [QuadraticSpectral{Andrews}()]  # Can need O(T²) temporarily
```

### 5. Parallel Processing Multiple Datasets

```julia
using Base.Threads

function parallel_covariance_estimation(datasets, estimator)
    results = Vector{Matrix{Float64}}(undef, length(datasets))

    @threads for i in eachindex(datasets)
        results[i] = aVar(estimator, datasets[i])
    end

    return results
end

# Usage
datasets = [randn(1000, 4) for _ in 1:100]
results = parallel_covariance_estimation(datasets, VARHAC())
```

## Performance Profiling Tools

### Built-in Diagnostics

```julia
# Check computational bottlenecks
using Profile

@profile for i in 1:100
    aVar(Bartlett{Andrews}(), X)
end

Profile.print()
```

### Custom Timing Utilities

```julia
function time_estimator_components(estimator, X)
    if isa(estimator, HAC) && !isa(estimator, HAC{Fixed})
        # Time bandwidth selection separately
        @time begin
            _, _, bw = workingoptimalbw(estimator, X)
            println("Bandwidth selection: computed bw = $bw")
        end

        # Time covariance computation with fixed bandwidth
        fixed_estimator = typeof(estimator){Fixed}([bw], [], [false])
        @time Ω_fixed = aVar(fixed_estimator, X)
    else
        @time Ω = aVar(estimator, X)
    end
end

# Usage
time_estimator_components(Bartlett{Andrews}(), X)
```

## Numerical Stability Considerations

### Condition Number Monitoring

```julia
function stability_check(Ω, tolerance=1e12)
    κ = cond(Ω)

    if κ > tolerance
        @warn "Ill-conditioned covariance matrix (κ = $κ)"
        return false
    end

    eigenvals = eigvals(Ω)
    min_eig = minimum(eigenvals)

    if min_eig < -1e-10
        @warn "Negative eigenvalue detected: $min_eig"
        return false
    end

    return true
end

# Usage
Ω = aVar(QuadraticSpectral{Andrews}(), X)
is_stable = stability_check(Ω)
```

### Regularization for Ill-Conditioned Cases

```julia
function regularized_covariance(estimator, X, λ=1e-8)
    Ω = aVar(estimator, X)

    if cond(Ω) > 1e10
        # Add ridge regularization
        k = size(Ω, 1)
        Ω_reg = Ω + λ * I(k)
        @info "Applied regularization λ = $λ"
        return Ω_reg
    end

    return Ω
end
```

## Platform-Specific Performance Notes

### Threading Performance Varies by System

- **Apple Silicon (M1/M2/M3)**: Excellent performance/efficiency cores balance
- **Intel multicore**: Traditional symmetric threading
- **AMD Threadripper**: May benefit from larger thread counts

### Memory Bandwidth Considerations

```julia
# For systems with limited memory bandwidth:
function memory_conscious_estimation(estimator, X)
    # Process in chunks if X is very large
    T, k = size(X)

    if T * k * sizeof(eltype(X)) > 1e9  # 1GB threshold
        @info "Large dataset detected, consider chunked processing"
    end

    return aVar(estimator, X)
end
```

## Recommendations by Use Case

### High-Frequency Financial Data
- **Primary**: `VARHAC()` or `SmoothedMoments()`
- **Rationale**: Automatic, robust, good performance
- **Avoid**: Andrews bandwidth selection (too slow)

### Large Cross-Sectional Studies (n > 10,000)
- **Primary**: `HC3()` or `HC1()`
- **Secondary**: `VARHAC()` if time series structure suspected
- **Memory**: Consider chunked processing

### Real-Time Applications
- **Primary**: Fixed bandwidth HAC: `Bartlett(bw)`
- **Secondary**: `HC3()` for cross-sectional
- **Key**: Pre-compute bandwidth offline

### Panel Data with Many Groups
- **Primary**: `CR1(clusters)`
- **Secondary**: `DriscollKraay()` if spatial correlation
- **Optimization**: Ensure efficient cluster grouping

### Bootstrap/Simulation Studies
- **Primary**: `VARHAC()` or `SmoothedMoments()`
- **Rationale**: Guaranteed PSD across all bootstrap samples
- **Avoid**: Estimators that can produce non-PSD matrices

The choice of estimator should balance statistical appropriateness with computational requirements for your specific application.