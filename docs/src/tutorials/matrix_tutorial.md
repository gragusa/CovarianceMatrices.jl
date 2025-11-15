# Matrix Interface Tutorial

This tutorial demonstrates how to use CovarianceMatrices.jl with the matrix interface for direct covariance matrix estimation. This approach is ideal when you have moment conditions or residuals and need to compute robust covariance matrices directly.

## Basic Workflow

The matrix interface follows this general pattern:

1. Prepare your data matrix (moment conditions, residuals, etc.)
2. Choose an appropriate estimator
3. Compute the covariance matrix using `aVar()`
4. Extract standard errors if needed

```julia
using CovarianceMatrices, LinearAlgebra, Random
Random.seed!(123)  # For reproducibility
```

## Example 1: Time Series with Serial Correlation

Let's start with simulated time series data that exhibits serial correlation:

```julia
# Generate AR(1) time series with multiple variables
T = 500
ρ = 0.6
k = 3

# Initialize
X = zeros(T, k)
ε = randn(T, k)

# Generate AR(1) process: X_t = ρ * X_{t-1} + ε_t
for t in 2:T
    X[t, :] = ρ * X[t-1, :] + ε[t, :]
end

println("Generated $(T)×$(k) time series with AR(1) coefficient ρ = $ρ")
```

### HAC Estimation

For time series data, HAC estimators account for both heteroskedasticity and autocorrelation:

```julia
# 1. Bartlett kernel with Andrews bandwidth selection
bart_andrews = Bartlett{Andrews}()
Ω_bart_andrews = aVar(bart_andrews, X)
println("Bartlett (Andrews): trace = $(round(tr(Ω_bart_andrews), digits=3))")

# 2. Bartlett kernel with fixed bandwidth
bart_fixed = Bartlett(5)  # bandwidth = 5
Ω_bart_fixed = aVar(bart_fixed, X)
println("Bartlett (fixed): trace = $(round(tr(Ω_bart_fixed), digits=3))")

# 3. Parzen kernel with Newey-West bandwidth
parzen_nw = Parzen{NeweyWest}()
Ω_parzen_nw = aVar(parzen_nw, X)
println("Parzen (Newey-West): trace = $(round(tr(Ω_parzen_nw), digits=3))")

# 4. Quadratic Spectral kernel
qs_andrews = QuadraticSpectral{Andrews}()
Ω_qs_andrews = aVar(qs_andrews, X)
println("Quadratic Spectral: trace = $(round(tr(Ω_qs_andrews), digits=3))")
```

### VARHAC Estimation (Recommended for Automatic Approach)

VARHAC eliminates the need for bandwidth selection by fitting a VAR model:

```julia
# Basic VARHAC with AIC selection
varhac_aic = VARHAC()  # Defaults: AIC, SameLags(8)
Ω_varhac_aic = aVar(varhac_aic, X)
println("VARHAC (AIC): trace = $(round(tr(Ω_varhac_aic), digits=3))")

# VARHAC with BIC selection
varhac_bic = VARHAC(:bic)
Ω_varhac_bic = aVar(varhac_bic, X)
println("VARHAC (BIC): trace = $(round(tr(Ω_varhac_bic), digits=3))")

# Check selected lag orders
println("AIC selected lags: ", order(varhac_aic))
println("BIC selected lags: ", order(varhac_bic))
```

### Smoothed Moments Estimation

Smith's smoothed moments method provides automatic positive semi-definiteness:

```julia
# Uniform kernel (induces Bartlett HAC)
sm_uniform = SmoothedMoments(UniformSmoother())
Ω_sm_uniform = aVar(sm_uniform, X)
println("Smoothed Moments (Uniform): trace = $(round(tr(Ω_sm_uniform), digits=3))")

# Triangular kernel (induces Parzen HAC)
sm_triangular = SmoothedMoments(TriangularSmoother())
Ω_sm_triangular = aVar(sm_triangular, X)
println("Smoothed Moments (Triangular): trace = $(round(tr(Ω_sm_triangular), digits=3))")

# Fixed bandwidth
sm_fixed = SmoothedMoments(UniformSmoother(), 8.0)
Ω_sm_fixed = aVar(sm_fixed, X)
println("Smoothed Moments (Fixed): trace = $(round(tr(Ω_sm_fixed), digits=3))")
```

## Example 2: Cross-Sectional Data with Heteroskedasticity

For cross-sectional data without serial correlation, use HC/HR estimators:

```julia
# Generate cross-sectional data with heteroskedasticity
N = 200
x = randn(N, 2)
β = [1.0, -0.5]
# Heteroskedastic errors: variance depends on x
σ² = exp.(0.5 * x[:, 1])
ε = σ² .* randn(N)
y = x * β + ε

# Compute residuals (this would be your moment conditions)
residuals = reshape(y - x * β, N, 1)

# HC estimators
hc_estimators = [HC0(), HC1(), HC2(), HC3(), HC4(), HC5()]
for (i, hc) in enumerate(hc_estimators)
    Ω_hc = aVar(hc, residuals)
    println("HC$(i-1): σ̂² = $(round(Ω_hc[1,1], digits=4))")
end
```

## Example 3: Clustered Data

For data with cluster correlation, use CR estimators:

```julia
# Generate clustered data
G = 20  # Number of clusters
n_per_cluster = 10
N = G * n_per_cluster

# Cluster indicators
clusters = repeat(1:G, inner=n_per_cluster)

# Generate clustered data
cluster_effects = randn(G)
individual_effects = randn(N)
y_clustered = cluster_effects[clusters] + 0.5 * individual_effects

# Residuals from some model
residuals_clustered = reshape(y_clustered .- mean(y_clustered), N, 1)

# CR estimators
cr_estimators = [CR0(clusters), CR1(clusters), CR2(clusters), CR3(clusters)]
for (i, cr) in enumerate(cr_estimators)
    Ω_cr = aVar(cr, residuals_clustered)
    println("CR$(i-1): σ̂² = $(round(Ω_cr[1,1], digits=4))")
end
```

## Example 4: Panel Data with Driscoll-Kraay

For panel data with both cross-sectional and time dependence:

```julia
# Panel dimensions
T_panel = 50
N_panel = 30
total_obs = T_panel * N_panel

# Create panel identifiers
time_ids = repeat(1:T_panel, outer=N_panel)
unit_ids = repeat(1:N_panel, inner=T_panel)

# Generate panel data with both dimensions of dependence
# Time effects
time_effects = cumsum(randn(T_panel))[time_ids] * 0.5
# Unit effects
unit_effects = randn(N_panel)[unit_ids] * 0.3
# Individual noise
noise = randn(total_obs) * 0.2

panel_data = time_effects + unit_effects + noise
residuals_panel = reshape(panel_data .- mean(panel_data), total_obs, 1)

# Driscoll-Kraay estimator
dk_estimator = DriscollKraay(Bartlett{Andrews}(), tis=time_ids, iis=unit_ids)
Ω_dk = aVar(dk_estimator, residuals_panel)
println("Driscoll-Kraay: σ̂² = $(round(Ω_dk[1,1], digits=4))")
```

## Example 5: EWC Estimation

The Equal Weighted Cosine estimator provides a non-parametric alternative:

```julia
# EWC with different numbers of basis functions
for B in [5, 10, 15]
    ewc_est = EWC(B)
    Ω_ewc = aVar(ewc_est, X)  # Using AR(1) data from Example 1
    println("EWC (B=$B): trace = $(round(tr(Ω_ewc), digits=3))")
end
```

## Advanced Usage: Custom Options and Diagnostics

### Prewhitening for HAC Estimators

Prewhitening can improve finite-sample performance of HAC estimators:

```julia
# HAC with prewhitening
Ω_prewhite = aVar(bart_andrews, X; prewhite=true)
Ω_no_prewhite = aVar(bart_andrews, X; prewhite=false)

println("Bartlett without prewhitening: $(round(tr(Ω_no_prewhite), digits=3))")
println("Bartlett with prewhitening: $(round(tr(Ω_prewhite), digits=3))")
```

### Bandwidth Diagnosis for HAC

```julia
# Extract optimal bandwidth
_, _, bw = workingoptimalbw(bart_andrews, X)
println("Optimal Andrews bandwidth: $(round(bw, digits=2))")

# Compare with rule-of-thumb
bw_newey_west = 4 * (T/100)^(2/9)
println("Newey-West rule-of-thumb: $(round(bw_newey_west, digits=2))")
```

### Memory and Performance Considerations

```julia
using BenchmarkTools

# Compare performance of different estimators
println("Performance comparison on $(size(X)) matrix:")

# Fast estimators
@btime aVar(HC3(), $X)
@btime aVar(VARHAC(), $X)

# Medium complexity
@btime aVar(Bartlett(5), $X)
@btime aVar(SmoothedMoments(), $X)

# More computationally intensive
@btime aVar(Parzen{Andrews}(), $X)
```

## Summary of Recommendations

### When to Use Each Estimator

1. **HC/HR (HC0-HC5)**: Cross-sectional data with heteroskedasticity only
2. **HAC (Bartlett, Parzen, etc.)**: Time series with both heteroskedasticity and autocorrelation
3. **VARHAC**: Time series when you want automatic bandwidth selection and guaranteed PSD
4. **Smoothed Moments**: Time series when you want automatic PSD with traditional HAC-like results
5. **CR (CR0-CR3)**: Data with cluster correlation
6. **Driscoll-Kraay**: Panel data with spatial and temporal correlation
7. **EWC**: Financial time series or when other methods are sensitive to specification

### Performance Tips

1. **For large datasets**: Use `HC3()` or `VARHAC()` for best performance
2. **For automatic approaches**: Use `VARHAC()` or `SmoothedMoments()`
3. **For maximum compatibility**: Use `Bartlett{Andrews}()` or `Parzen{NeweyWest}()`
4. **For guaranteed PSD**: Use `VARHAC()`, `SmoothedMoments()`, or `EWC()`

### Common Pitfalls

1. **Wrong estimator choice**: Using HC for time series data or HAC for cross-sectional data
2. **Bandwidth sensitivity**: HAC results can be sensitive to bandwidth choice
3. **Small sample bias**: Consider HC2/HC3 over HC0/HC1 for small samples
4. **Cluster size**: CR estimators require sufficient cluster size for good properties

This tutorial covers the essential usage patterns for the matrix interface. The next tutorial will show how to integrate these estimators with GLM.jl for econometric modeling.