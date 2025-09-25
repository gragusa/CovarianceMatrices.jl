# Equal Weighted Cosine (EWC) Estimator

The Equal Weighted Cosine (EWC) estimator provides a non-parametric approach to robust covariance estimation using cosine basis functions. It offers an alternative to traditional HAC estimators by using a different basis for approximating the spectral density.

## Mathematical Foundation

### Spectral Density Approach

The EWC estimator approximates the long-run covariance matrix through the spectral density at frequency zero. Instead of using kernels and bandwidths like HAC estimators, EWC uses a cosine basis expansion.

The estimator computes the covariance matrix using:

```math
\hat{\Omega}_{EWC} = \sum_{j=1}^{B} w_j \cos\left(\frac{\pi j \tau}{T}\right) \hat{\Gamma}_j
```

where:
- $B$ is the number of basis functions
- $w_j$ are equal weights: $w_j = \frac{1}{B}$
- $\tau$ represents the lag structure
- $\hat{\Gamma}_j$ are sample autocovariances

### Key Features

1. **Non-parametric**: No assumptions about specific correlation structure
2. **Equal weighting**: All basis functions receive equal weight
3. **Automatic bandwidth**: Number of basis functions $B$ serves as bandwidth parameter
4. **Computational efficiency**: Direct basis function evaluation

## Core Type

```@docs
EWC
```

## Usage Examples

### Basic Usage

```julia
using CovarianceMatrices, Random
Random.seed!(123)

# Generate time series data with serial correlation
T = 200
X = zeros(T, 3)
for t in 2:T
    X[t, :] = 0.6 * X[t-1, :] + randn(3)
end

# EWC with different numbers of basis functions
ewc_5 = EWC(5)
ewc_10 = EWC(10)
ewc_20 = EWC(20)

Ω_5 = aVar(ewc_5, X)
Ω_10 = aVar(ewc_10, X)
Ω_20 = aVar(ewc_20, X)

println("EWC(5) trace: $(round(tr(Ω_5), digits=3))")
println("EWC(10) trace: $(round(tr(Ω_10), digits=3))")
println("EWC(20) trace: $(round(tr(Ω_20), digits=3))")
```

### Comparison with HAC Estimators

```julia
using CovarianceMatrices, Random, LinearAlgebra
Random.seed!(123)

# Generate AR(1) process
T = 300
ρ = 0.7
X = zeros(T, 2)
for t in 2:T
    X[t, :] = ρ * X[t-1, :] + randn(2) * 0.5
end

# Compare EWC with HAC estimators
estimators = [
    ("EWC(5)", EWC(5)),
    ("EWC(10)", EWC(10)),
    ("EWC(15)", EWC(15)),
    ("Bartlett(5)", Bartlett(5)),
    ("Bartlett(10)", Bartlett(10)),
    ("VARHAC", VARHAC())
]

println("Estimator\\t\\tTrace\\tCondition Number")
for (name, est) in estimators
    Ω = aVar(est, X)
    tr_val = tr(Ω)
    cond_val = cond(Ω)
    println("$(rpad(name, 15))\\t$(round(tr_val, digits=3))\\t$(round(cond_val, digits=1))")
end
```

### GLM Integration

```julia
using GLM, DataFrames, CovarianceMatrices

# Generate regression data with serial correlation
T = 150
x1 = randn(T)
x2 = cumsum(randn(T)) / sqrt(T)  # Random walk regressor

# Serially correlated errors
ε = zeros(T)
for t in 2:T
    ε[t] = 0.5 * ε[t-1] + randn()
end

y = 1.0 .+ 0.8 * x1 .- 1.2 * x2 .+ ε

df = DataFrame(y=y, x1=x1, x2=x2)
model = lm(@formula(y ~ x1 + x2), df)

# Compare standard errors
se_classical = stderror(model)
se_ewc5 = stderror(EWC(5), model)
se_ewc10 = stderror(EWC(10), model)
se_ewc20 = stderror(EWC(20), model)

results = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Classical = round.(se_classical, digits=4),
    EWC_5 = round.(se_ewc5, digits=4),
    EWC_10 = round.(se_ewc10, digits=4),
    EWC_20 = round.(se_ewc20, digits=4)
)

println(results)
```

## Basis Function Selection

### Choosing the Number of Basis Functions

The number of basis functions $B$ is the key tuning parameter:

```julia
function basis_function_analysis(X)
    T, k = size(X)

    # Try different numbers of basis functions
    B_values = [3, 5, 8, 10, 15, 20, 25]

    println("B\\tTrace\\t\\tCondition\\tMin Eigenvalue")

    for B in B_values
        if B < T  # Ensure B < T for numerical stability
            ewc = EWC(B)
            Ω = aVar(ewc, X)

            eigenvals = eigvals(Ω)
            trace_val = tr(Ω)
            cond_val = cond(Ω)
            min_eig = minimum(eigenvals)

            println("$B\\t$(round(trace_val, digits=3))\\t\\t$(round(cond_val, digits=1))\\t\\t$(round(min_eig, digits=6))")
        end
    end
end

# Apply to data
basis_function_analysis(X)
```

### Rule of Thumb

General guidelines for selecting $B$:

```julia
function suggest_basis_functions(T)
    if T <= 50
        return 3
    elseif T <= 100
        return 5
    elseif T <= 200
        return 8
    elseif T <= 500
        return 12
    else
        return min(20, T ÷ 25)
    end
end

# Example usage
T = size(X, 1)
suggested_B = suggest_basis_functions(T)
println("For T=$T, suggested B = $suggested_B")
```

## Theoretical Properties

### Advantages of EWC

1. **Flexibility**: Cosine basis can approximate many spectral shapes
2. **Simplicity**: Single tuning parameter (number of basis functions)
3. **Computational efficiency**: Direct evaluation without iterative bandwidth selection
4. **Positive semi-definiteness**: Guaranteed under proper implementation

### Comparison with HAC

| Feature | EWC | HAC |
|---------|-----|-----|
| Tuning parameter | Number of basis functions | Bandwidth |
| Selection method | Cross-validation or rules-of-thumb | Andrews, Newey-West |
| Computational cost | Low (direct evaluation) | Medium (kernel computation) |
| Flexibility | High (cosine basis) | Medium (kernel choice) |

## Advanced Usage

### Cross-Validation for Basis Selection

```julia
function cross_validate_ewc(X, B_candidates, folds=5)
    T, k = size(X)
    fold_size = T ÷ folds
    cv_errors = Dict()

    for B in B_candidates
        errors = Float64[]

        for fold in 1:folds
            # Create train/test split
            test_start = (fold - 1) * fold_size + 1
            test_end = min(fold * fold_size, T)

            train_idx = setdiff(1:T, test_start:test_end)
            test_idx = test_start:test_end

            if length(train_idx) > B + 10  # Ensure sufficient data
                # Fit on training data
                X_train = X[train_idx, :]
                ewc_train = EWC(B)
                Ω_train = aVar(ewc_train, X_train)

                # Simple prediction error (conceptual)
                X_test = X[test_idx, :]
                pred_error = norm(cov(X_test) - Ω_train[1:k, 1:k])
                push!(errors, pred_error)
            end
        end

        if !isempty(errors)
            cv_errors[B] = mean(errors)
        end
    end

    # Find best B
    best_B = argmin(cv_errors)

    println("Cross-validation results:")
    for (B, error) in sort(collect(cv_errors))
        marker = B == best_B ? " ←" : ""
        println("B=$B: CV Error = $(round(error, digits=4))$marker")
    end

    return best_B
end

# Example usage (if sufficient data)
if size(X, 1) > 100
    B_candidates = [5, 8, 10, 12, 15]
    optimal_B = cross_validate_ewc(X, B_candidates)
    println("\\nOptimal B: $optimal_B")
end
```

### Sensitivity Analysis

```julia
function ewc_sensitivity_analysis(X, base_B=10)
    # Test sensitivity around base choice
    B_range = max(3, base_B-3):min(25, base_B+5)

    results = []
    base_ewc = EWC(base_B)
    base_Ω = aVar(base_ewc, X)

    println("B\\tTrace Ratio\\tFrobenius Distance")

    for B in B_range
        if B != base_B
            ewc = EWC(B)
            Ω = aVar(ewc, X)

            trace_ratio = tr(Ω) / tr(base_Ω)
            frob_dist = norm(Ω - base_Ω, 2)

            println("$B\\t$(round(trace_ratio, digits=3))\\t\\t$(round(frob_dist, digits=3))")
            push!(results, (B, trace_ratio, frob_dist))
        end
    end

    return results
end

# Apply sensitivity analysis
sensitivity_results = ewc_sensitivity_analysis(X, 10)
```

## Performance Considerations

### Computational Complexity

```julia
using BenchmarkTools

function ewc_performance_comparison()
    # Generate test data of different sizes
    sizes = [100, 200, 500, 1000]

    println("Performance Comparison:")
    println("T\\tEWC(10)\\t\\tBartlett(5)\\tVARHAC")

    for T in sizes
        X_perf = randn(T, 3)

        # Time different estimators
        time_ewc = @belapsed aVar(EWC(10), $X_perf)
        time_bart = @belapsed aVar(Bartlett(5), $X_perf)
        time_var = @belapsed aVar(VARHAC(), $X_perf)

        println("$T\\t$(round(time_ewc*1000, digits=1))ms\\t\\t$(round(time_bart*1000, digits=1))ms\\t\\t$(round(time_var*1000, digits=1))ms")
    end
end

ewc_performance_comparison()
```

### Memory Usage

EWC is generally memory-efficient:

```julia
function ewc_memory_analysis(T_values, k=3)
    println("Memory scaling analysis:")
    println("T\\tApprox Memory (EWC)")

    for T in T_values
        # Rough memory estimate for EWC computation
        memory_estimate = T * k * sizeof(Float64) * 2  # Input matrix + temporaries
        memory_mb = memory_estimate / (1024^2)

        println("$T\\t$(round(memory_mb, digits=2)) MB")
    end
end

ewc_memory_analysis([100, 500, 1000, 5000])
```

## When to Use EWC

### ✅ Recommended for:

1. **Exploratory analysis**: When you want to avoid bandwidth selection
2. **Moderate serial correlation**: Works well for AR-type processes
3. **Computational efficiency**: When speed is important
4. **Robustness**: When you want an alternative to HAC

### ❌ Consider alternatives for:

1. **Very short time series** (T < 30): May not have enough data
2. **Highly persistent processes**: HAC with automatic bandwidth may be better
3. **Well-understood correlation structure**: Parametric methods may be more efficient

## Integration with Other Estimators

### Model Selection Framework

```julia
function compare_robust_estimators(X, model=nothing)
    # Compare EWC with other approaches
    estimators = [
        ("EWC(5)", EWC(5)),
        ("EWC(10)", EWC(10)),
        ("Bartlett-Andrews", Bartlett{Andrews}()),
        ("VARHAC", VARHAC()),
        ("Smoothed Moments", SmoothedMoments())
    ]

    results = Dict()

    for (name, est) in estimators
        try
            if model === nothing
                Ω = aVar(est, X)
                results[name] = (tr(Ω), cond(Ω))
            else
                se = stderror(est, model)
                results[name] = se
            end
        catch e
            results[name] = "Error: $e"
        end
    end

    return results
end

# Usage
comparison_results = compare_robust_estimators(X)
for (name, result) in comparison_results
    if isa(result, Tuple)
        println("$name: Trace=$(round(result[1], digits=3)), Cond=$(round(result[2], digits=1))")
    else
        println("$name: $result")
    end
end
```

## Best Practices

### Selection Guidelines

1. **Start conservative**: Begin with B = 5-10 for moderate sample sizes
2. **Check stability**: Verify results are not too sensitive to B choice
3. **Compare with HAC**: Use as robustness check against HAC estimators
4. **Monitor eigenvalues**: Ensure all eigenvalues are positive

### Diagnostic Checks

```julia
function ewc_diagnostics(ewc_estimator, X)
    Ω = aVar(ewc_estimator, X)
    eigenvals = eigvals(Ω)

    println("EWC Diagnostics:")
    println("Basis functions (B): $(ewc_estimator.B)")
    println("Matrix size: $(size(Ω))")
    println("Condition number: $(round(cond(Ω), digits=1))")
    println("Min eigenvalue: $(round(minimum(eigenvals), digits=6))")
    println("Max eigenvalue: $(round(maximum(eigenvals), digits=3))")

    if minimum(eigenvals) < -1e-10
        println("⚠️  Negative eigenvalues detected!")
    end

    if cond(Ω) > 1e12
        println("⚠️  Ill-conditioned matrix!")
    end

    return eigenvals
end

# Example usage
eigenvals = ewc_diagnostics(EWC(10), X)
```

## Applications

### Financial Time Series

EWC is particularly useful for financial applications with moderate persistence.

### Macro-econometric Models

Works well for macroeconomic time series with unknown correlation structures.

### Signal Processing

The cosine basis makes EWC natural for signal processing applications.

## References

**Theoretical Background:**
- The EWC estimator draws from spectral analysis and harmonic analysis literature
- Related to kernel density estimation and non-parametric spectral estimation

**Applications:**
- Financial econometrics applications
- Time series analysis with unknown correlation structures
- Robustness studies comparing different covariance estimators

**Computational Methods:**
- Efficient computation of cosine basis functions
- Numerical linear algebra for covariance matrix estimation