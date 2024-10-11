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
   - VarHAC (Experimental)
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

One might want to calculate a variance estimator when the regression (or some other model) is fit "manually". Below is an example of how this can be accomplished.

```julia
using Random, StableRNGs, LinearAlgebra

rng = StableRNGs.StableRNG(666111)
n, k = (100, 5)
K = k+1
# Fake regression data
X = [ones(n) randn(rng, 100, k)]
y = randn(rng, n) 
# OLS coefficients
b   = X\y
# OLS residuals
res = y .- X*b
# momentmatrix
momentmatrix = X.*res
# ∝ Jacobian of moment conditions: ∂ ∑g(xᵢ,β)/n / ∂ β
B = inv(X'X/n) 
# Estimate of the Asymptotic VARiance of ∑g(xᵢ,β)/√n
A = aVar(Bartlett(3), momentmatrix)
## Estimate of asymptotic distribution of √β
Σ = (B*A*B)
## Standard errors 
sqrt.(diag(Σ./n))
## Standard errors (with correction)
sqrt.(diag(Σ./(n-k))

# Compare with GLM
sdterror(Bartlett(3), lm(X,y); dofadjust=false)

sdterror(Bartlett(3), lm(X,y); dofadjust=true)
```


## Advanced Usage


`CovarianceMatrices.jl` is designed to be flexible and extensible. It can be used to estimate the asymptotic variance of custom estimators by defining the `invpseudohessian` and `momentmatrix` methods. 

Below a simple example of how to extend `CovarianceMatrices.jl` for a `Probit` model:

```julia
using Optim, CovarianceMatrices, Distributions, ForwardDiff, LinearAlgebra
using RDatasets

hmda  = dataset("Ecdat", "Hdma")

X = [ones(size(hmda, 1)) hmda.DIR hmda.LVR hmda.CCS]
y = ifelse.(hmda.Deny.=="yes", 1, 0)


struct Probit{T<:AbstractMatrix, V<:AbstractVector}
    X::T
    y::V
    coef
    function Probit(X::T, y::V) where {T, V}
        new{T, V}(X, y, Array{Float64}(undef, size(X, 2)))
    end
end

# Define the log-likelihood function
function (loglik::Probit)(β::AbstractVector)
    X, y = loglik.X, loglik.y
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

# Fit the model
ℓ = Probit(X, y)
res = optimize(x->-ℓ(x), X\y, BFGS(); autodiff = :forward)
ℓ.coef .= Optim.minimizer(res)

# Extend CovarianceMatrices.jl methods
function CovarianceMatrices.invpseudohessian(loglik::Probit)
    -inv(ForwardDiff.hessian(loglik, loglik.coef)) * length(loglik.y)
end

function CovarianceMatrices.momentmatrix(loglik::Probit)
    X, y = loglik.X, loglik.y
    η = X * loglik.coef
    ϕ = pdf.(Normal(), η)
    Φ = cdf.(Normal(), η)
    ((1.0 ./ Φ) .* y .- (1.0 ./ (1 .- Φ)) .* (1 .- y)) .* ϕ .* X
end

# Calculate standard errors and robust standard errors
n = length(ℓ.y)
H⁻¹ = CovarianceMatrices.invpseudohessian(ℓ)
s = CovarianceMatrices.momentmatrix(ℓ)
Ω = aVar(HC0(), s)
Σ = H⁻¹ * Ω * H⁻¹

standard_errors = sqrt.(diag(H⁻¹) ./ n)
robust_standard_errors = sqrt.(diag(Σ) ./ n)

```

For more detailed examples and advanced usage, please refer to the full documentation.

## Contributing

Contributions to CovarianceMatrices.jl are welcome! Please feel free to submit issues and pull requests on our GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Performance

CovarianceMatrices.jl is designed for high performance which might turnout to be useful in those cases where the asymptotic variance of estimators need to be computed repeatedly, e.g. for bootstrap inference. 

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
