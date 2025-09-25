# VARHAC: Vector Autoregression HAC Estimation

VARHAC (Vector Autoregression Heteroskedasticity and Autocorrelation Consistent) provides a data-driven alternative to traditional HAC estimation that eliminates the need for bandwidth selection and automatically ensures positive semi-definiteness.

## Mathematical Foundation

### The VAR Approach

Instead of choosing kernels and bandwidths, VARHAC fits a Vector Autoregression model to the moment conditions:

```math
g_t = c + \sum_{j=1}^p A_j g_{t-j} + \varepsilon_t
```

where:
- $g_t$ are the moment conditions at time $t$
- $c$ is a vector of constants
- $A_j$ are coefficient matrices
- $p$ is the lag length
- $\varepsilon_t$ are i.i.d. errors with covariance $\Sigma_\varepsilon$

### Long-run Covariance Computation

The long-run covariance matrix is computed as:

```math
\hat{\Omega}_{VARHAC} = \left(I - \sum_{j=1}^p \hat{A}_j\right)^{-1} \hat{\Sigma}_\varepsilon \left(I - \sum_{j=1}^p \hat{A}_j'\right)^{-1}
```

This formula comes from the Wold representation of the VAR process and represents the spectral density at frequency zero.

### Advantages over Traditional HAC

1. **No bandwidth selection**: The lag length $p$ is selected via information criteria
2. **Automatic PSD**: The formula guarantees positive semi-definiteness
3. **Data adaptive**: Captures complex serial correlation patterns
4. **Computational efficiency**: No kernel computations required

## Core Type and Constructors

```@docs
VARHAC
```

## Lag Selection Methods

### Information Criteria

The lag length $p$ is typically selected using information criteria:

```@docs
AICSelector
BICSelector
FixedSelector
```

**AIC vs BIC Trade-offs:**
- **AIC**: Tends to select longer lags, captures more dynamics
- **BIC**: Tends to select shorter lags, more parsimonious
- **Fixed**: Use when you have strong prior knowledge

### Lag Strategies

Different strategies are available for specifying how lags are searched:

```@docs
SameLags
FixedLags
AutoLags
DifferentOwnLags
```

## Basic Usage Examples

### Default VARHAC

The simplest usage relies on sensible defaults:

```julia
using CovarianceMatrices, Random
Random.seed!(123)

# Generate VAR(2) data
T = 300
A1 = [0.5 0.1; 0.2 0.3]
A2 = [0.2 -0.1; 0.1 0.4]
Σ = [1.0 0.3; 0.3 1.0]

X = zeros(T, 2)
for t in 3:T
    X[t, :] = A1 * X[t-1, :] + A2 * X[t-2, :] + rand(MvNormal(Σ))
end

# Estimate with defaults (AIC, SameLags(8))
varhac_default = VARHAC()
Ω_default = aVar(varhac_default, X)

# Check selected lag order
println("Selected lag order: ", order(varhac_default))
println("Trace of covariance: ", round(tr(Ω_default), digits=3))
```

### Comparing Selection Methods

```julia
# Different selectors
varhac_aic = VARHAC(:aic)
varhac_bic = VARHAC(:bic)

Ω_aic = aVar(varhac_aic, X)
Ω_bic = aVar(varhac_bic, X)

println("AIC selected lags: ", order(varhac_aic))
println("BIC selected lags: ", order(varhac_bic))
println("AIC trace: ", round(tr(Ω_aic), digits=3))
println("BIC trace: ", round(tr(Ω_bic), digits=3))
```

### Different Lag Strategies

```julia
# Same maximum lags for all variables
varhac_same = VARHAC(AICSelector(), SameLags(12))

# Fixed lag length (no selection)
varhac_fixed = VARHAC(FixedLags(4))

# Automatic lag selection based on sample size
varhac_auto = VARHAC(AICSelector(), AutoLags())

# For bivariate data: different own lags
varhac_diff = VARHAC(AICSelector(), DifferentOwnLags([3, 5]))

estimators = [
    ("Same Lags", varhac_same),
    ("Fixed Lags", varhac_fixed),
    ("Auto Lags", varhac_auto),
    ("Different Own Lags", varhac_diff)
]

for (name, est) in estimators
    Ω = aVar(est, X)
    lags = isa(est.strategy, FixedLags) ? [est.strategy.maxlag] : order(est)
    println("$name: lags = $lags, trace = $(round(tr(Ω), digits=3))")
end
```

## Advanced Features

### Information Criteria Diagnostics

VARHAC stores the computed information criteria, allowing for post-estimation diagnostics:

```julia
# Fit VARHAC model
varhac_diag = VARHAC(:aic)
Ω = aVar(varhac_diag, X)

# Extract information criteria
aics = AICs(varhac_diag)
bics = BICs(varhac_diag)

println("AIC values: ", round.(aics, digits=2))
println("BIC values: ", round.(bics, digits=2))
println("AIC optimal lag: ", order_aic(varhac_diag))
println("BIC optimal lag: ", order_bic(varhac_diag))

# Plot information criteria (if Plots.jl available)
# using Plots
# plot([aics bics], label=["AIC" "BIC"], title="Information Criteria")
```

### Maximum Lag Selection

```julia
# Access maximum lags for different strategies
println("SameLags(10): ", maxlags(VARHAC(SameLags(10))))
println("FixedLags(5): ", maxlags(VARHAC(FixedLags(5))))

# AutoLags requires data dimensions
auto_strategy = VARHAC(AutoLags())
auto_max_lags = maxlags(auto_strategy, size(X, 1), size(X, 2))
println("AutoLags for T=$(size(X,1)), N=$(size(X,2)): ", auto_max_lags)
```

## Practical Guidelines

### When to Use VARHAC

✅ **Good for:**
- Time series with complex serial correlation patterns
- When you want to avoid bandwidth selection
- Applications requiring guaranteed positive semi-definiteness
- Exploratory analysis where robustness is key
- Multivariate time series with cross-correlation

❌ **Consider alternatives for:**
- Very short time series (T < 50)
- Cross-sectional data (use HC estimators)
- When you need to match specific HAC kernel results
- Real-time applications with computational constraints

### Selector Choice Guidelines

```julia
# Conservative approach: use BIC
varhac_conservative = VARHAC(:bic)

# Flexible approach: use AIC
varhac_flexible = VARHAC(:aic)

# When you know the true lag structure
varhac_known = VARHAC(FixedLags(true_lag_length))

# Let the data decide based on sample size
varhac_adaptive = VARHAC(AutoLags())
```

### Sample Size Considerations

```julia
function sample_size_analysis(T_values)
    results = []
    for T in T_values
        # Generate simple AR(1) data
        X_sample = zeros(T, 2)
        for t in 2:T
            X_sample[t, :] = 0.5 * X_sample[t-1, :] + randn(2)
        end

        # Fit VARHAC
        ve = VARHAC()
        Ω = aVar(ve, X_sample)
        selected_lag = order(ve)[1]

        push!(results, (T, selected_lag, tr(Ω)))
    end

    println("Sample Size Analysis:")
    println("T\tSelected Lag\tTrace")
    for (T, lag, trace) in results
        println("$T\t$lag\t\t$(round(trace, digits=3))")
    end
end

# Run analysis
sample_size_analysis([50, 100, 200, 500, 1000])
```

## Comparison with Traditional HAC

### Performance Comparison

```julia
using BenchmarkTools

# Generate test data
T = 500
X_test = cumsum(randn(T, 3), dims=1)  # Random walk

# VARHAC
varhac = VARHAC()
@btime aVar($varhac, $X_test)

# Traditional HAC with bandwidth selection
bartlett_andrews = Bartlett{Andrews}()
@btime aVar($bartlett_andrews, $X_test)

# Traditional HAC with fixed bandwidth
bartlett_fixed = Bartlett(8)
@btime aVar($bartlett_fixed, $X_test)
```

### Accuracy Comparison

```julia
# Generate AR(2) process
T = 400
true_A1 = [0.6 0.1; 0.2 0.4]
true_A2 = [0.1 -0.1; 0.05 0.2]

X_ar2 = zeros(T, 2)
for t in 3:T
    X_ar2[t, :] = true_A1 * X_ar2[t-1, :] + true_A2 * X_ar2[t-2, :] + randn(2)
end

# Compare different estimators
estimators_comp = [
    ("VARHAC (AIC)", VARHAC(:aic)),
    ("VARHAC (BIC)", VARHAC(:bic)),
    ("Bartlett-Andrews", Bartlett{Andrews}()),
    ("Parzen-Andrews", Parzen{Andrews}()),
    ("Newey-West", Bartlett{NeweyWest}())
]

println("Method\t\t\tTrace\tMin Eigenval\tCondition")
for (name, est) in estimators_comp
    Ω = aVar(est, X_ar2)
    eigenvals = eigvals(Ω)
    println("$(rpad(name, 15))\t$(round(tr(Ω), digits=3))\t$(round(minimum(eigenvals), digits=4))\t\t$(round(cond(Ω), digits=1))")
end
```

## Troubleshooting and Diagnostics

### Common Issues and Solutions

**1. Selected lag too high/low**
```julia
# Check information criteria progression
varhac_check = VARHAC()
aVar(varhac_check, X)

aics = AICs(varhac_check)
min_idx = argmin(aics)
println("AIC minimum at lag: $min_idx")
println("AIC values: ", round.(aics[max(1, min_idx-2):min(end, min_idx+2)], digits=3))

# Try different strategy if needed
if min_idx > 8
    varhac_longer = VARHAC(SameLags(15))
    Ω_longer = aVar(varhac_longer, X)
end
```

**2. Numerical issues with matrix inversion**
```julia
# Check condition number
varhac_num = VARHAC()
Ω_num = aVar(varhac_num, X)

if cond(Ω_num) > 1e12
    println("Warning: Ill-conditioned covariance matrix")
    # Try shorter lags
    varhac_short = VARHAC(FixedLags(2))
    Ω_short = aVar(varhac_short, X)
end
```

**3. Sensitivity to lag specification**
```julia
# Lag sensitivity analysis
function lag_sensitivity(X, max_lag=10)
    traces = Float64[]
    for p in 1:max_lag
        ve = VARHAC(FixedLags(p))
        Ω = aVar(ve, X)
        push!(traces, tr(Ω))
    end

    println("Lag\tTrace")
    for (p, trace) in enumerate(traces)
        println("$p\t$(round(trace, digits=3))")
    end

    return traces
end

traces = lag_sensitivity(X, 8)
```

## Integration with GLM

VARHAC works seamlessly with GLM.jl for time series regression:

```julia
using GLM, DataFrames

# Create time series regression data
T_glm = 200
trend = collect(1:T_glm) / T_glm
y_ts = 2.0 .+ 1.5 * trend + cumsum(randn(T_glm) * 0.5)  # With unit root
x_ts = trend + randn(T_glm) * 0.2

df_ts = DataFrame(y=y_ts, x=x_ts)

# Fit regression
model_ts = lm(@formula(y ~ x), df_ts)

# Get VARHAC standard errors
se_varhac = stderror(VARHAC(), model_ts)
se_classical = stderror(model_ts)

println("Classical SEs: ", round.(se_classical, digits=4))
println("VARHAC SEs: ", round.(se_varhac, digits=4))
println("Ratio (VARHAC/Classical): ", round.(se_varhac ./ se_classical, digits=2))
```

## Advanced Configuration

### Custom Type Specifications

```julia
# Use Float32 for memory efficiency with large datasets
varhac_f32 = VARHAC(T=Float32)

# Different precision for different applications
varhac_f64 = VARHAC(T=Float64)  # Default
```

### Integration with Other Estimators

```julia
# Use VARHAC as benchmark for other methods
function compare_with_varhac(X, other_estimators)
    # Reference: VARHAC
    Ω_varhac = aVar(VARHAC(), X)

    println("Estimator\t\tTrace Ratio\tFrobenius Distance")
    println("VARHAC (reference)\t1.000\t\t0.000")

    for (name, est) in other_estimators
        Ω_est = aVar(est, X)
        trace_ratio = tr(Ω_est) / tr(Ω_varhac)
        frob_dist = norm(Ω_est - Ω_varhac, 2)
        println("$(rpad(name, 20))\t$(round(trace_ratio, digits=3))\t\t$(round(frob_dist, digits=3))")
    end
end

other_estimators = [
    ("Bartlett-Andrews", Bartlett{Andrews}()),
    ("Parzen-NeweyWest", Parzen{NeweyWest}()),
    ("Smoothed Moments", SmoothedMoments())
]

compare_with_varhac(X, other_estimators)
```

VARHAC provides a robust, automatic, and theoretically sound approach to HAC estimation that eliminates many of the subjective choices required by traditional methods while guaranteeing positive semi-definite results.

## References

- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". *Handbook of Statistics*, 15, 291-341.
- Dufour, J.M., Pelletier, D., and Renault, E. (2006). "Short run and long run causality in time series: inference". *Journal of Econometrics*, 132(2), 337-362.
- Pelletier, D. (2011). "Regime switching for dynamic correlations". *Journal of Econometrics*, 131(1-2), 445-473.