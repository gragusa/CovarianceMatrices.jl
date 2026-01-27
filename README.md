# CovarianceMatrices.jl

[![CI](https://github.com/gragusa/CovarianceMatrices.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/CovarianceMatrices.jl/actions/workflows/ci.yml) [![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)


`CovarianceMatrices.jl` is a Julia package for robust covariance matrix estimation. It provides consistent estimates of the long-run covariance matrix of random processes, which is crucial for conducting valid statistical inference about the parameters of econometric and statistical models.

The package offers several classes of estimators to handle different data structures and dependence patterns:

1. **HAC** (Heteroskedasticity and Autocorrelation Consistent)
   - **Kernel-based** estimators (`Bartlett`, `Parzen`, `QuadraticSpectral`, `Truncated`)
   - **EWC** (Exponentially Weighted Covariance)
   - **Smoothed moments** estimators
   - **VARHAC** (VAR-based HAC with data-driven lag selection)
2. **HC** (Heteroskedasticity Consistent) - for cross-sectional data with independent observations
3. **CR** (Cluster Robust) - for data with clustering structure
4. **Driscoll-Kraay** - for panel data with cross-sectional dependence

## Installation

```julia
using Pkg
Pkg.add("CovarianceMatrices")
```

## As a GLM.jl Extension

`CovarianceMatrices.jl` seamlessly extends methods from `StatsBase.jl` and `GLM.jl`, enabling robust standard error calculations for generalized linear models.

Here are some basic examples demonstrating how to use `CovarianceMatrices.jl` to obtain robust standard errors with `GLM.jl` models:

```julia
using RDatasets
using GLM
using CovarianceMatrices

df = dataset("plm", "Grunfeld")
model = glm(@formula(Inv ~ Value + Capital), df, Normal(), IdentityLink())

# Calculate HAC standard errors using Bartlett kernel with optimal bandwidth
# Andrews bandwidth selection
vcov_hac_andrews = vcov(Bartlett{Andrews}(), model)

# Newey-West bandwidth selection
vcov_hac_nw = vcov(Bartlett{NeweyWest}(), model)

# Calculate heteroskedasticity-robust (HC) standard errors
vcov_hc = vcov(HC1(), model)

# Calculate cluster-robust standard errors (clustered by firm)
vcov_cr = vcov(CR1(df.Firm), model)

# Calculate Driscoll-Kraay standard errors for panel data
# (accounts for cross-sectional dependence and heteroskedasticity)
vcov_dk = vcov(DriscollKraay(Bartlett(5), tis=df.year, iis=df.firm), model)
```

For **heteroskedasticity-robust variance** estimation, `HC0`, `HC1`, `HC2`, `HC3`, `HC4`, and `HC5` are the standard estimator types commonly used in the econometric literature:
- `HC0` is the basic White (1980) estimator
- `HC1` applies a degrees-of-freedom correction: `n/(n-k)`
- `HC2`, `HC3`, `HC4`, and `HC5` are refined variations that adjust for leverage points and improve small-sample performance. These are, however, only defined for (generalized) linear models.



For **serially correlated errors**, HAC estimators account for both heteroskedasticity and autocorrelation in the error terms. Common kernel choices include `Bartlett`, `Parzen`, `QuadraticSpectral`, and `Truncated`, each with its own weighting scheme based on lag distance. The bandwidth can be specified directly (e.g., `Parzen(3)` uses a bandwidth of 3), or selected optimally using data-driven methods such as Andrews (1991) or Newey-West (1994). Another nonparametric estimator is the `Smoothed` estimator. `VARHAC` is a parametric estimator. If the correlation comes from the presence of clusters, then `CR` methods are provided.   

## As an Extension for Custom Statistical Models

`CovarianceMatrices.jl` provides a unified API for calculating variance estimators in custom models. 

We distinguish between two broad classes of estimators:

- **M-Estimators**: Models estimated by extremum estimation (e.g., Maximum Likelihood, Quasi-ML)
- **GMM Estimators**: Models estimated via the Generalized Method of Moments

Since these two classes have different requirements for variance estimation, the API provides a clear separation. Custom models should inherit from one of the following abstract types:

- `MLikeModel`: For M-estimators
- `GMMLikeModel`: For GMM estimators

For both model types, we provide two variance forms:
- **Information Form**: The standard variance estimator used when the model is correctly specified (based on the Fisher Information Matrix for MLE)
- **Misspecified Form**: A robust sandwich variance estimator that remains consistent even if the model is misspecified (also known as the Huber-White or robust variance estimator)

### Model Interface

Third-party estimators should implement the following methods:

**Required methods:**
- `CovarianceMatrices.momentmatrix(model)`: Return the moment matrix or score function contributions (n × k matrix)
- `StatsAPI.coef(model)`: Return parameter estimates (k-vector)
- `StatsAPI.nobs(model)`: Return number of observations

**Optional methods (depending on variance form):**

- `CovarianceMatrices.hessian_objective(model)`: Return the objective Hessian (k × k matrix) - required for Misspecified form with GMM
- `CovarianceMatrices.jacobian_momentfuns(model)`: Return the jecobian of the momentfuns (m × m matrix) - required for GMM

### Important: Scaling Convention

**The `hessian_objective()` and `jacobian_momentfunction()` methods should return unscaled matrices**, i.e., sums over observations.

## Examples

### M-like Models

Consider a simple Probit model estimated via Maximum Likelihood Estimation (MLE). MLE is a special case of M-estimator, since the estimator solves the first-order conditions of the optimization problem. We define a custom type `SimpleProbit` that implements the required interface for M-like models.

```julia
using CovarianceMatrices
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using Optimization
using OptimizationOptimJL

# Define Probit model with type hierarchy
struct SimpleProbit <: CovarianceMatrices.MLikeModel
    y::Vector{Int}
    X::Matrix{Float64}
    β::Vector{Float64}
    fitted_probs::Vector{Float64}
    fitted_densities::Vector{Float64}
end

# Probit log-likelihood function
function probit_loglik(β, y, X)
    Xβ = X * β
    probs = cdf.(Normal(), Xβ)
    # Avoid log(0) with small epsilon
    probs = clamp.(probs, 1e-15, 1 - 1e-15)
    return sum(y .* log.(probs) .+ (1 .- y) .* log.(1 .- probs))
end

# Negative log-likelihood for optimization (we minimize)
function neg_loglik(β, p)
    y, X = p
    return -probit_loglik(β, y, X)
end

# Constructor that performs MLE optimization
function SimpleProbit(y::Vector{Int}, X::Matrix{Float64})
    # Initial values (e.g., from OLS)
    β_init = X \ y

    # Set up optimization problem
    optf = OptimizationFunction(neg_loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, β_init, (y, X))

    # Solve using BFGS from Optim.jl
    sol = solve(prob, BFGS())
    β_opt = sol.u

    # Compute fitted values
    Xβ = X * β_opt
    fitted_probs = cdf.(Normal(), Xβ)
    fitted_densities = pdf.(Normal(), Xβ)

    return SimpleProbit(y, X, β_opt, fitted_probs, fitted_densities)
end

# Implement required interface
StatsAPI.coef(m::SimpleProbit) = m.β
StatsAPI.nobs(m::SimpleProbit) = length(m.y)

# Score functions (for MLE, this is the gradient of log-likelihood)
function CovarianceMatrices.momentmatrix(m::SimpleProbit)
    residuals = m.y .- m.fitted_probs
    weights = m.fitted_densities ./ (m.fitted_probs .* (1 .- m.fitted_probs) .+ 1e-15)
    return m.X .* (residuals .* weights)
end

# Objective Hessian (analytical second derivative of log-likelihood)
function CovarianceMatrices.hessian_objective(m::SimpleProbit)
    Xβ = m.X * m.β
    qᵢ = 2*m.y .- 1  # +1 for y=1, -1 for y=0
    Φ = cdf.(Normal(), qᵢ.*Xβ)
    ϕ = pdf.(Normal(), qᵢ.*Xβ)
    λ₁ = qᵢ .* (ϕ ./ Φ)  # Inverse Mills ratio for y=1 and y=0
    w = -(λ₁ .* (λ₁ .+ Xβ))
    -(m.X' * Diagonal(w) * m.X)
end

# Example usage
n = 1000
rng = StableRNG(1234)
X = [ones(n) randn(rng, n) randn(rng, n)]
β_true = [0.5, 1.0, -0.5]
y_latent = X * β_true + randn(rng, n)
y = Int.(y_latent .> 0)

# Estimate model
model = SimpleProbit(y, X)

# Now you can use both variance forms:
vcov_info = vcov(HC1(), Information(), model)          # Fisher Information-based
vcov_robust = vcov(Bartlett(3), Misspecified(), model) # Robust sandwich estimator
```

### GMM-like Models

This code demonstrates the use of the `CovarianceMatrices.jl` package to perform Generalized Method of Moments (GMM) estimation using a custom-defined `LinearGMM` type for instrumental variables regression.

```julia
using CovarianceMatrices
using LinearAlgebra
using Statistics
using StatsBase
using Random
using Test

# Simple IV/GMM model structure
struct LinearGMM{T, V, K} <: CovarianceMatrices.GMMLikeModel
    data::T          # Data tuple (y, X, Z)
    beta_fs::V       # First-step estimates
    beta::V          # Final GMM estimates (mutable via copy!)
    v::K             # Variance estimator for weighting matrix
end

# Implement CovarianceMatrices.jl interface
StatsAPI.coef(m::LinearGMM) = m.beta
StatsAPI.nobs(m::LinearGMM) = length(m.data.y)

# Moment conditions: Z'(y - X*β)
function CovarianceMatrices.momentmatrix(p::LinearGMM, beta)
    y, X, Z = p.data
    return Z .* (y .- X * beta)
end

function CovarianceMatrices.momentmatrix(p::LinearGMM)
    return CovarianceMatrices.momentmatrix(p, coef(p))
end


## Estimate the parameters using two-step GMM with identity weighting matrix in the first step
function LinearGMM(data; v::CovarianceMatrices.AbstractAsymptoticVarianceEstimator = HR0())
    y, X, Z = data

    ## First Step GMM with identity weighting matrix
    W = pinv(Z' * Z)
    beta_fs = (X' * Z) * W * (Z' * X) \ (X' * Z) * W * (Z' * y)
    gmm = LinearGMM(data, beta_fs, similar(beta_fs), v)

    ## Second Step: Use optimal weighting matrix
    M = CovarianceMatrices.momentmatrix(gmm, beta_fs)
    Omega = aVar(v, M)
    W_opt = pinv(Omega)
    beta_opt = (X' * Z) * W_opt * (Z' * X) \ (X' * Z) * W_opt * (Z' * y)
    copy!(gmm.beta, beta_opt)

    return gmm
end

# Objective Hessian for GMM (used in Misspecified form)
function CovarianceMatrices.hessian_objective(p::LinearGMM)
    y, X, Z = p.data
    M = CovarianceMatrices.momentmatrix(p, coef(p))
    Omega = aVar(p.v, M; scale = false)
    H = -(X' * Z) * pinv(Omega) * (Z' * X)
    return H
end

# Objective Hessian for GMM (used in Misspecified form)
function CovarianceMatrices.jacobian_momentfunction(p::LinearGMM)
    y, X, Z = p.data
    G = -Z'* X 
    return G
end

## Data is a named tuple with y (dependent variable),
## X (endogenous regressors), and Z (instruments)
Random.seed!(123)
n = 100
data = (
    y = randn(n),
    X = [ones(n) randn(n)],
    Z = [ones(n) randn(n) randn(n)]
)

model = LinearGMM(data)

## Standard variance estimator (assumes correct specification)
V1 = vcov(HR0(), Information(), model)

## Misspecified/robust variance (uses the Hessian)
V2 = vcov(HR0(), Misspecified(), model)

## Estimate the model with HAC-based optimal weighting matrix
model_hac = LinearGMM(data; v = Bartlett(5))

# Information-form variance with HAC
V3 = vcov(Bartlett(5), Information(), model_hac)

## Sandwich variance (robust to moment condition misspecification)
V4 = vcov(Bartlett(5), Misspecified(), model_hac)

```

## Performance

`CovarianceMatrices.jl` is designed for high performance, particularly useful in applications where covariance estimators need to be computed repeatedly, such as bootstrap-based inference, simulation studies, or iterative estimation procedures.

To give an idea of the performance, below is a quick comparison with the `sandwich` package in R for computing HAC covariance matrices.

### Julia (`CovarianceMatrices.jl`)

```julia
using BenchmarkTools, CovarianceMatrices
Z = randn(10000, 10)
@btime aVar($(Bartlett{Andrews}()), $Z; prewhite = true)
```

```shell
467.125 μs (128 allocations: 4.33 MiB)
```

### R (`sandwich`)

```R
library(sandwich)
library(microbenchmark)
Z <- matrix(rnorm(10000*10), 10000, 10)
microbenchmark("Bartlett/Andrews" = {lrvar(Z, type = "Andrews", kernel = "Bartlett", adjust=FALSE)})
```

```shell
Unit: milliseconds
              expr      min       lq     mean   median       uq      max neval
 Bartlett/Andrews 59.56402 60.76790 63.85169 61.47827 68.73355 82.26539   100
```

**CovarianceMatrices.jl is approximately 130× faster than the R implementation for this benchmark.**

## Contributing

Contributions to CovarianceMatrices.jl are welcome! Please feel free to submit issues and pull requests on our [GitHub repository](https://github.com/gragusa/CovarianceMatrices.jl).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
