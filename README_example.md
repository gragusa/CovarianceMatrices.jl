# CovarianceMatrices.jl - New Unified API

## Overview

The new API provides a unified interface for variance estimation across different model types and assumptions. It uses Julia's type system for clean dispatch and implements numerically stable algorithms.

## Key Features

- **Type-safe variance forms**: `Information`, `Robust`, `CorrectlySpecified`, `Misspecified`
- **Automatic model detection**: Auto-selects appropriate variance form based on model dimensions
- **Numerically stable**: Uses factorizations instead of matrix inversions
- **Duck-typed integration**: Simple interface for third-party estimators

## Quick Example: Probit Model

```julia
using CovarianceMatrices
using LinearAlgebra, Statistics, StatsBase, Random

# Your custom model struct
struct MyProbitModel
    y::Vector{Int}
    X::Matrix{Float64}
    β::Vector{Float64}
    # ... other fields
end

# Implement the CovarianceMatrices.jl interface
StatsBase.coef(m::MyProbitModel) = m.β

function CovarianceMatrices.momentmatrix(m::MyProbitModel)
    # Return n × m matrix of moment conditions/score contributions
    # ...
end

function CovarianceMatrices.objective_hessian(m::MyProbitModel)
    # Return k × k Hessian matrix (for Information form)
    # ...
end

function CovarianceMatrices.jacobian(m::MyProbitModel)
    # Return m × k Jacobian matrix (for Robust form)
    # ...
end

# Fit your model
model = fit_probit(y, X)  # Your fitting routine

# Use the new API
V_info = vcov_new(HC1(), Information(), model)     # Information matrix
V_robust = vcov_new(HC3(), Robust(), model)       # Robust sandwich
V_auto = vcov_new(HC1(), model; form=:auto)       # Automatic selection

# Standard errors
se = stderror_new(HC1(), Information(), model)
```

## Variance Forms

### M-like Models (exactly identified, m = k)

- **`Information()`**: Uses V = H⁻¹ (assumes correct specification)
- **`Robust()`**: Uses V = G⁻¹ΩG⁻ᵀ (robust to misspecification)

### GMM-like Models (overidentified, m > k)

- **`CorrectlySpecified()`**: Uses V = (G'Ω⁻¹G)⁻¹ (optimal under correct specification)
- **`Misspecified()`**: Uses V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹ (robust to misspecification)

## Automatic Form Selection

When using `form=:auto`:
- **M-like models** (m = k): Defaults to `Robust()` for safety
- **GMM-like models** (m > k): Defaults to `CorrectlySpecified()` for efficiency

## Matrix-Based API

For advanced use or when you have matrices directly:

```julia
# Your matrices
Z = your_moment_matrix    # n × m
G = your_jacobian        # m × k
H = your_hessian         # k × k

# Compute variance
V = vcov_new(HC1(), Information(), Z; objective_hessian=H)
V = vcov_new(Bartlett(5), Robust(), Z; jacobian=G)
```

## Supported Estimators

All existing variance estimators work with the new API:

- **Heteroskedasticity-robust**: `HC0()`, `HC1()`, `HC2()`, `HC3()`, `HC4()`, `HC5()`
- **HAC estimators**: `Bartlett(bw)`, `Parzen(bw)`, `QuadraticSpectral(bw)`
- **Clustering**: `CR0(groups)`, `CR1(groups)`, `CR2(groups)`, `CR3(groups)`
- **Panel data**: `DriscollKraay(kernel, time_var, group_var)`

## Migration from Old API

The old API remains fully functional. New features use the `_new` suffix:

```julia
# Old API (still works)
V_old = vcov(HC1(), model)

# New API (recommended)
V_new = vcov_new(HC1(), Information(), model)
```

## Complete Working Example

Here's a full working example with a simple Probit model:

```julia
using CovarianceMatrices, LinearAlgebra, Statistics, StatsBase, Random

# Helper functions
normal_cdf(x) = 0.5 * (1 + sign(x) * sqrt(1 - exp(-2x^2/π)))
normal_pdf(x) = exp(-x^2/2) / sqrt(2π)

# Simple Probit implementation
struct SimpleProbit
    y::Vector{Int}
    X::Matrix{Float64}
    β::Vector{Float64}
    fitted_probs::Vector{Float64}
    fitted_densities::Vector{Float64}
end

function fit_simple_probit(y, X)
    β = (X'X) \\ (X'y)  # Simple starting values
    Xβ = X * β
    probs = normal_cdf.(Xβ)
    densities = normal_pdf.(Xβ)
    return SimpleProbit(y, X, β, probs, densities)
end

# CovarianceMatrices.jl interface
StatsBase.coef(m::SimpleProbit) = m.β

function CovarianceMatrices.momentmatrix(m::SimpleProbit)
    residuals = m.y .- m.fitted_probs
    weights = m.fitted_densities ./ (m.fitted_probs .* (1 .- m.fitted_probs) .+ 1e-15)
    return m.X .* (residuals .* weights)
end

function CovarianceMatrices.objective_hessian(m::SimpleProbit)
    weights = m.fitted_densities.^2 ./ (m.fitted_probs .* (1 .- m.fitted_probs) .+ 1e-15)
    return (m.X' * Diagonal(weights) * m.X) / length(m.y)
end

# Generate data and fit model
Random.seed!(123)
n, k = 500, 3
X = [ones(n) randn(n, k-1)]
β_true = [0.5, 1.0, -0.8]
y = Int.(rand(n) .< normal_cdf.(X * β_true))

model = fit_simple_probit(y, X)

# Compute variances with new API
V_hc1 = vcov_new(HC1(), Information(), model)
V_hc3 = vcov_new(HC3(), Information(), model)
V_bartlett = vcov_new(Bartlett(3), Information(), model)

# Standard errors
se_hc1 = stderror_new(HC1(), Information(), model)
se_hc3 = stderror_new(HC3(), Information(), model)

println("HC1 standard errors: ", round.(se_hc1, digits=4))
println("HC3 standard errors: ", round.(se_hc3, digits=4))
```

## Benefits of the New API

1. **Type Safety**: Catch errors at compile time rather than runtime
2. **Clarity**: Explicit variance forms make assumptions clear
3. **Flexibility**: Easy to extend for new model types
4. **Performance**: Numerically stable algorithms with optimal dispatch
5. **Consistency**: Unified interface across all estimator types

The new API is ready for production use and will be the recommended approach going forward.