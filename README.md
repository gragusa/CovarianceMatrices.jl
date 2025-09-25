# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=master)](https://travis-ci.org/gragusa/CovarianceMatrices.jl) [![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=master) [![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=master)

`CovarianceMatrices.jl` is a Julia package for robust covariance matrix estimation. It provides consistent estimates of the long-run covariance matrix of random processes, which is crucial for conducting inference about the parameters of statistical models.

## Installation

```julia
using Pkg
Pkg.add("CovarianceMatrices")
```


## Features

The package offers several classes of estimators:

1. **HAC** (Heteroskedasticity and Autocorrelation Consistent)
   - Kernel-based
   - EWC (Exponentially Weighted Covariance)
   - Smoothed (Experimental)
   - VarHAC (TBA)
2. **HC** (Heteroskedasticity Consistent)
3. **CR** (Cluster Robust)
4. **Driscoll-Kraay**

## Quick Examples
`CovarianceMatrices.jl` extends methods from `StatsBase.jl` and `GLM.jl` extending calculation of standard error for generalized linear models.

Here are some basic examples of how to use `CovarianceMatrices.jl` for obtaining standard errors with `GLM.jl` models:

```julia
using RDatasets
df = dataset("plm", "Grunfeld")
model = glm(@formula(Inv~Value+Capital), df, Normal(), IdentityLink())
# Calculate HAC standard errors using Bartlet Kernel with the optimal
# Bandwidth a' la Andrews
vcov_hac = vcov(Bartlett{Andrews}(), model)
# Calculate heteroskedasticity-robust (HC) standard errors
vcov_hc = vcov(HC1(), model)
# Calculate cluster-robust standard errors
vcov_cr = vcov(CR1(df.Firm), model)
# Calculate Driscoll-Kraay standard errors (Bartlett kernel)
vcov_dk = vcov(DriscollKraay(Bartlett(5), tis=df.year, iis=df.firm), model)
```

## Unified API

`CovarianceMatrices.jl` now features a unified API that provides a consistent interface for both Maximum Likelihood and GMM estimators. The new API supports:

### Variance Forms

The unified API uses just two variance forms that adapt based on model type:

- **Information Form**:
  * MLE: Fisher Information Matrix V = H⁻¹
  * GMM: Efficient GMM V = (G'Ω⁻¹G)⁻¹
- **Misspecified Form**:
  * MLE: Robust sandwich V = G⁻¹ΩG⁻ᵀ
  * GMM: Robust GMM V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹

### Type Hierarchy

The package uses a type hierarchy for proper method dispatch:

- `MLikeModel`: For Maximum Likelihood estimators
- `GMMLikeModel`: For GMM estimators

Both model types support both variance forms, which automatically adapt their behavior.

### Estimator Hierarchy

The package features a semantic estimator hierarchy for clarity:

- `AbstractAsymptoticVarianceEstimator`: Root type for all covariance estimators
  - `Uncorrelated`: For iid errors (use instead of HC0 for MLE/GMM models)
  - `Correlated`: For various correlation patterns
    - `HAC` estimators: Time series correlation (Bartlett, Parzen, etc.)
    - `CR` estimators: Cluster correlation (CR0, CR1, CR2, CR3)
    - `EWC`, `VARHAC`, `DriscollKraay`, `SmoothedMoments`: Specialized estimators
  - `HR` estimators: Heteroskedasticity-robust (for linear models with conditional heteroskedasticity)

**Usage Guidelines:**
```julia
# For MLE/GMM with iid errors - use Uncorrelated() for semantic clarity
se_uncorr = stderror(Uncorrelated(), mle_model)

# For linear models with conditional heteroskedasticity - use HR/HC as before
se_robust = stderror(HC1(), linear_model)

# For time series or clustered data - use Correlated subtypes
se_hac = stderror(Bartlett{NeweyWest}(), time_series_model)
se_cluster = stderror(CR1(cluster_ids), panel_model)
```

### Usage Example

```julia
using CovarianceMatrices

# For MLE models
vcov_info = vcov(HC1(), Information(), model)     # Fisher Information
vcov_robust = vcov(HC1(), Misspecified(), model)  # Robust sandwich

# For GMM models
vcov_efficient = vcov(HR0(), Information(), model)  # Efficient GMM
vcov_robust_gmm = vcov(HR0(), Misspecified(), model)  # Robust GMM
```

### Model Interface

Third-party estimators should implement:

- `CovarianceMatrices.momentmatrix(model)`: Return moment matrix/score functions
- `CovarianceMatrices.score(model)`: Return Jacobian matrix (required for Robust and GMM forms)
- `CovarianceMatrices.objective_hessian(model)`: Return objective Hessian (optional, for Misspecified form)
- `StatsBase.coef(model)`: Return parameter estimates
- `StatsBase.nobs(model)`: Return number of observations

## Advanced Usage

`CovarianceMatrices.jl` is designed to be flexible and extensible. The examples below show both the legacy interface and the new unified API.

Here's how to implement a Probit model using the new unified API:

```julia
using CovarianceMatrices
using FiniteDifferences
using LinearAlgebra
using Statistics
using StatsBase

# Define Probit model with type hierarchy
struct SimpleProbit <: CovarianceMatrices.MLikeModel
    y::Vector{Int}
    X::Matrix{Float64}
    β::Vector{Float64}
    fitted_probs::Vector{Float64}
    fitted_densities::Vector{Float64}
end

# Implement required interface
StatsBase.coef(m::SimpleProbit) = m.β
StatsBase.nobs(m::SimpleProbit) = length(m.y)

# Score functions (for MLE, this is the gradient of log-likelihood)
function CovarianceMatrices.momentmatrix(m::SimpleProbit)
    residuals = m.y .- m.fitted_probs
    weights = m.fitted_densities ./ (m.fitted_probs .* (1 .- m.fitted_probs) .+ 1e-15)
    return m.X .* (residuals .* weights)
end

# Jacobian matrix (negative Fisher Information for MLE)
function CovarianceMatrices.score(m::SimpleProbit)
    weights = m.fitted_densities.^2 ./ (m.fitted_probs .* (1 .- m.fitted_probs) .+ 1e-15)
    return -(m.X' * Diagonal(weights) * m.X) / length(m.y)
end

# Objective Hessian (Fisher Information Matrix)
function CovarianceMatrices.objective_hessian(m::SimpleProbit)
    return -score(m)
end

# Now you can use both variance forms:
vcov_info = vcov(HC1(), Information(), model)          # Fisher Information
vcov_robust = vcov(Bartlett(3), Misspecified(), model) # Robust sandwich
```

## GMM-like models

This code demonstrates the use of the `CovarianceMatrices.jl` package to perform Generalized Method of Moments (GMM) estimation using a custom-defined `LinearGMM` type.

```julia
using CovarianceMatrices
using LinearAlgebra
using Statistics
using StatsBase
using Random
using Test

# Simple IV model structure
struct LinearGMM{T, V, K} <: CovarianceMatrices.GMMLikeModel
    data::T
    beta_fs::V
    beta::V      # Estimated coefficients
    v::K
end

# Implement CovarianceMatrices.jl interface
StatsBase.coef(m::LinearGMM) = m.beta
StatsBase.nobs(m::LinearGMM) = length(m.data.y)

function CovarianceMatrices.momentmatrix(p::LinearGMM, beta)
    y, X, Z = p.data
    Z.*(y .- X*beta)
end

function CovarianceMatrices.momentmatrix(p::LinearGMM)
    y, X, Z = p.data
    Z.*(y .- X*coef(p))
end

function CovarianceMatrices.score(p::LinearGMM)
    y, X, Z = p.data
    return -(Z' * X)./nobs(p)
end

## Constructor - We estimate the parameters
## using the TSLS initial matrix.
function LinearGMM(data; v::CovarianceMatrices.AbstractAsymptoticVarianceEstimator = HR0())
    y, X, Z = data
    ## First Step GMM
    W = pinv(Z'Z)
    #Main.@infiltrate
    beta_fs = (X'*Z)*W*(Z'X)\(X'*Z)*W*(Z'y)
    gmm = LinearGMM(data, beta_fs, similar(beta_fs), v)
    ## Second Step
    M = CovarianceMatrices.momentmatrix(gmm, beta_fs)
    Omega = aVar(v, M)
    beta_fs = (X'*Z)*Omega*(Z'X)\(X'*Z)*Omega*(Z'y)
    copy!(gmm.beta, beta_fs)
    return gmm
end

function CovarianceMatrices.objective_hessian(p::LinearGMM)
    y, X, Z = data
    M = CovarianceMatrices.momentmatrix(p, coef(p))
    Omega = aVar(p.v, M)
    n = nobs(p)
    H = -(X'Z/n)*pinv(Omega)*(Z'X/n)
    return H
end

## Usage
## Assume data is a named tuple with y, X, and the instruments Z.


model = LinearGMM(data)

coef(model)

## Usual variance estimator
V1 = vcov(HR0(), Information(), model)

## This use the hessian
V2 = vcov(HR0(), Misspecified(), model

## Etimate the model with a kernel second step optimal weighting
model = LinearGMM(data; v = Bartlett(5))
# Usual variance
V3 = vcov(Bartlett(3), Information(), model)

## Sandwich variance (robust to misspecification of the moment
V4 = vcov(Bartlett(3), Misspecified(), model)

```

## Performance

CovarianceMatrices.jl is designed for high performance, which might be useful in cases where the asymptotic variance of estimators needs to be computed repeatedly, e.g., for bootstrap inference.

Benchmark comparison with the `sandwich` package in R:

### Julia (`CovarianceMatrices.jl`)

```julia
using BenchmarkTools, CovarianceMatrices
Z = randn(10000, 10)
@btime aVar($(Bartlett{Andrews}()), $Z; prewhite = true)

```
```
467.125 μs (128 allocations: 4.33 MiB)
```

### R (`sandwich`)
```r
library(sandwich)
library(microbenchmark)
Z <- matrix(rnorm(10000*10), 10000, 10)
microbenchmark( "Bartlett/Newey" = {lrvar(Z, type = "Andrews", kernel = "Bartlett", adjust=FALSE)})
```

```
Unit: milliseconds
        expr    min      lq      mean     median   uq       max      neval
 Bartlett/Newey 59.56402 60.7679 63.85169 61.47827 68.73355 82.26539 100
```


## Contributing

Contributions to CovarianceMatrices.jl are welcome! Please feel free to submit issues and pull requests on our GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
