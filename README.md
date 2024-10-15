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

`CovarianceMatrices.jl` extends methods from `StatsBase.jl` and `GLM.jl`, providing a seamless replacement for standard error calculations in linear models.

## Quick Examples

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

## Advanced Usage

`CovarianceMatrices.jl` is designed to be flexible and extensible. It can be used to estimate the asymptotic variance of custom estimators by defining the `bread` and `momentmatrix` methods. 


Below is a simple example illustrating how to extend `CovarianceMatrices.jl` for calculating robust estimates of the asymptotic variance-covariance matrix of the parameters of a _probit_ model and of a `GMM` estimator.

### Probit model

This code implements a Probit regression model and it applies to the  `Hdma` dataset. The Probit model is defined in a custom Julia `struct`, which stores the design matrix `X` and binary outcome `y`, along with the coefficients. 

The code defines a log-likelihood function, and extends `momentmatrix` and `bread` methods to handle this model. 

The model is fitted using the BFGS optimization algorithm from the `Optim` package, and after fitting, standard errors and robust standard errors are calculated using various variance-covariance matrix estimators. 

```julia
using CovarianceMatrices
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using RDatasets

# ----------------------------------------------------------
hmda  = dataset("Ecdat", "Hdma")
X = [ones(size(hmda, 1)) hmda.DIR hmda.LVR hmda.CCS]
y = ifelse.(hmda.Deny.=="yes", 1, 0)
hmda.deny = y
# In `GLM.jl` notation, we estimate the following model
# glm(@formula(y~DIR+LVR+CCS), hmda, Binomal(), ProbitLink())
# ----------------------------------------------------------

# Define the Probit model
struct Probit{T<:AbstractMatrix, V<:AbstractVector}
    X::T
    y::V
    coef
    function Probit(X::T, y::V) where {T, V}
        new{T, V}(X, y, Array{Float64}(undef, size(X, 2)))
    end
end

StatsBase.coef(model::Probit) = model.coef

# Define the log-likelihood function
function (model::Probit)(β::AbstractVector)
    X, y = model.X, model.y
    n = length(y)
    @assert length(β) == size(X, 2) "Invalid dimensions"
    η = X * β
    ll = 0.0
    for i in 1:n
        p = cdf(Normal(), η[i])
        ll += y[i] * log(p) + (1 - y[i]) * log(1 - p)
    end
    return ll
end

# Extend `CovarianceMatrices.jl` `bread` method
function CovarianceMatrices.bread(model::Probit)
    ## Note: The loglikelihood does not divide by n
    ## so we do it 
    -inv(ForwardDiff.hessian(model, coef(model))) * length(model.y)
end

# Extend `CovarianceMatrices.jl` `momentmatrix` method
function CovarianceMatrices.momentmatrix(model::Probit, t)
    X, y = model.X, model.y
    η = X * t
    ϕ = pdf.(Normal(), η)
    Φ = cdf.(Normal(), η)
    ((1.0 ./ Φ) .* y .- (1.0 ./ (1 .- Φ)) .* (1 .- y)) .* ϕ .* X
end

function CovarianceMatrices.momentmatrix(model::Probit) 
    CovarianceMatrices.momentmatrix(model::Probit, coef(model))
end

# ----------------------------------------------------------
# Fit the model
# ----------------------------------------------------------
probit = Probit(X, y)
res = optimize(x->-probit(x), X\y, BFGS(); autodiff = :forward)
probit.coef .= Optim.minimizer(res)

# ----------------------------------------------------------
# Calculate standard errors and robust standard errors
# ----------------------------------------------------------
vcov(HC0(), probit)
stderror(HC0(), probit)
# Robust to correlation
stderror(Bartlett(4), probit)
```

### GMM

This code demonstrates the use of the `CovarianceMatrices.jl` package to perform Generalized Method of Moments (GMM) estimation using a custom-defined `GMMProblem` struct. 

The moment condition is 

$$
E[g(d_i, \theta)]=0,
$$

where 

$$g(d_i, \theta) := \begin{pmatrix} 
  d_i-\theta \\ 
  (d_i-\theta)^2-1 
  \end{pmatrix}.$$

The `momentmatrix` function returns the moment conditions based on the model's coefficients and data, while the `bread` function computes the GMM "bread" matrix, which involves the Jacobian of the moment conditions and the inverse of the weighting matrix. 

The GMM estimation proceeds in two steps: 

1. **First Step**: Uses the identity matrix as the initial weighting matrix and optimizes the GMM objective using the LBFGS algorithm.

3. **Second Step**: Computes an efficient weighting matrix from the first step's residuals using a heteroskedasticity-consistent estimator (HC1), then performs a second-step optimization for more efficient parameter estimates.

Finally, the code computes robust variance-covariance matrices for the estimates from both the first-step and second-step GMM models. This process ensures robust inference for the GMM estimates. 


```julia
using CovarianceMatrices
using ForwardDiff
using LinearAlgebra
using Optim
using StatsBase

struct GMMProblem{D, O}
    coef::Vector{Float64}
    d::D
    Ω::O
    Ω⁻::O
    function GMMProblem(d::D, Ω::O) where {D, O<:Union{Matrix, typeof(I)}}
        coef = Array{Float64}(undef, 1)
        new{D, O}(coef, d, Ω, pinv(Ω))
    end
end

StatsBase.coef(gmm::GMMProblem) = gmm.coef

function CovarianceMatrices.momentmatrix(gmm::GMMProblem)
    θ = coef(gmm)
    d = gmm.d
    [d[:,1].-θ[1]  (d[:,1].-θ[1]).^2.0.-1]
end

function CovarianceMatrices.bread(gmm::GMMProblem)
    gᵢ = CovarianceMatrices.momentmatrix(gmm)
    n, m = size(gᵢ)
    ∇g = ForwardDiff.jacobian(x->mean(CovarianceMatrices.momentmatrix(gmm,x); dims=1), coef(gmm))
    Ω⁻= gmm.Ω⁻
    (∇g'*Ω⁻*∇g)/Ω⁻'∇g
end

function CovarianceMatrices.momentmatrix(gmm::GMMProblem, θ)
    d = gmm.d
    [d[:,1].-θ[1]  (d[:,1].-θ[1]).^2.0.-1]
end

function (gmm::GMMProblem)(θ)
    gᵢ = CovarianceMatrices.momentmatrix(gmm, θ)
    gₙ = sum(gᵢ; dims=1)
    Ω⁻= gmm.Ω⁻
    0.5*first(gₙ*Ω⁻*gₙ')/size(gᵢ,1)
end

# ----------------------------------------------------------
# Fake data
# ----------------------------------------------------------
d = randn(100,1)
# ----------------------------------------------------------
# First Step
# Use identity matrix as weighting matrix
# ----------------------------------------------------------
firststep_gmm = GMMProblem(d, I)
first_step = Optim.optimize(firststep_gmm, [0.], LBFGS())
firststep_gmm.coef .= Optim.minimizer(first_step)
# ----------------------------------------------------------
# Second Step
# ----------------------------------------------------------
Ω = CovarianceMatrices.aVar(HC1(), momentmatrix(firststep_gmm))
efficient_gmm = GMMProblem(d, Ω)
second_step = Optim.optimize(efficient_gmm, coef(firststep_gmm), LBFGS())
efficient_gmm.coef .= Optim.minimizer(second_step)

vcov(HC0(), firststep_gmm)
vcov(HC0(), efficient_gmm)

## This covariance is asymptotically valid even if the estimator is 
## is not efficient.  
vcov(Bartlett(4), efficient_gmm)

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
681.166 μs (93 allocations: 3.91 MiB)
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


