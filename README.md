# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=master)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)
[![CovarianceMatrices](http://pkg.julialang.org/badges/CovarianceMatrices_0.6.svg)](http://pkg.julialang.org/detail/CovarianceMatrices&ver=0.6)
[![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=master)
[![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=master)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Installation

The package is registered on [METADATA](http::/github.com/JuliaLang/METADATA.jl), so to install
```julia
pkg> add CovarianceMatrices
```

---

## Introduction

This package provides types and methods useful to obtain consistent estimates of the long run covariance matrix of a random process.

Three classes of estimators are considered:

1. **HAC** - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West, 1994)
2. **HC**  - hetheroskedasticity consistent (White, 1982)
3. **CRVE** - cluster robust (Arellano, 1986; Bell, 2002)

The typical application of these estimators is to conduct robust inference about parameters of generalized linear models.

# Quick tour

## HAC (Heteroskedasticity and Autocorrelation Consistent)

Available kernel types are:

- `TruncatedKernel`
- `BartlettKernel`
- `ParzenKernel`
- `TukeyHanningKernel`
- `QuadraticSpectralKernel`

These types are subtypes of the abstract type `HAC`.

For example, `ParzenKernel(NeweyWest)` return an instance of `TruncatedKernel` parametrized by `NeweyWest`, the type that corresponds to the optimal bandwidth calculated following Newey and West (1994). Similarly, `ParzenKernel(Andrews)` corresponds to the optimal bandwidth obtained in Andrews (1991). If the bandwidth is known, it can be directly passed, i.e. `TruncatedKernel(2)`.

## Long-run variance of random vector

Consider testimating the long-run variance of a (p x 1) random vector X based on (T x 1) observations.

```julia
## X is (Txp)
CovarianceMatrices.variance(X, ParzenKernel())           ## Parzen Kernel with Optimal Bandwidth a lá Andrews
CovarianceMatrices.variance(X, ParzenKernel(NeweyWest))  ## Parzen Kernel with Optimal Bandwidth a lá Newey-West
CovarianceMatrices.variance(X, ParzenKernel(2))          ## Parzen Kernel with Bandwidth  = 2
```

Before calculating the variance the data can be prewhitened.

```julia
## X is (Txp)
CovarianceMatrices.variance(X, ParzenKernel(prewhiten=true))             ## Parzen Kernel with Optimal Bandwidth a lá Andrews
CovarianceMatrices.variance(X, ParzenKernel(NeweyWest, prewhiten=true))  ## Parzen Kernel with Optimal Bandwidth a lá Newey-West
CovarianceMatrices.variance(X, ParzenKernel(2, prewhiten=true))          ## Parzen Kernel with Bandwidth  = 2
```


### Long-run variance of the regression coefficient

In the regression context, the function `vcov` does all the work:
```julia
vcov(::DataFrameRegressionModel, ::HAC)
```

Consider the following artificial data (a regression with autoregressive error component):
```julia
using CovarianceMatrices
using DataFrames
using Random
Random.seed!(1)
n = 500
x = randn(n,5)
u = Array{Float64}(undef, 2*n)
u[1] = rand()
for j in 2:2*n
    u[j] = 0.78*u[j-1] + randn()
end


df = DataFrame()
df[:y] = y
for j in enumerate([:x1, :x2, :x3, :x4, :x5])
    df[j[2]] = x[:,j[1]]
end
```
Using the data in `df`, the coefficient of the regression can be estimated using `GLM`

```julia
lm1 = glm(@formula(y~x1+x2+x3+x4+x5), df, Normal(), IdentityLink())
```

To get a consistent estimate of the long run variance of the estimated coefficients using a Quadratic Spectral kernel with automatic bandwidth selection  _à la_ Andrews
```julia
vcov(lm1, QuadraticSpectralKernel(Andrews))
```
If one wants to estimate the long-time variance using the same kernel, but with a bandwidth selected _à la_ Newey-West
```julia
vcov(lm1, QuadraticSpectralKernel(NeweyWest))
```
The standard errors can be obtained by the `stderror` method
```julia
stderror(::DataFrameRegressionModel, ::HAC)
```
<!-- Sometime is useful to access the bandwidth selected by the automatic procedures. This can be done using the `optimalbw` method
```julia
optimalbw(NeweyWest, QuadraticSpectralKernel, lm1; prewhite = false)
optimalbw(Andrews, QuadraticSpectralKernel, lm1; prewhite = false)

## HC (Heteroskedastic consistent)

As in the HAC case, `vcov` and `stderr` are the main functions. They know get as argument the type of robust variance being sought
```julia
vcov(::DataFrameRegressionModel, ::HC)
```
Where `HC` is an abstract type with the following concrete types:

- `HC0`
    - This is Hal White (1982) robust variance estimator
- `HC1`
    - This is equal to `H0` multiplyed it by n/(n-p), where n is the sample size and p is the number of parameters in the model.
- `HC2`
    - A modification of HC0 that involves dividing the squared residual by 1-h, where h is the leverage for the case (Horn, Horn and Duncan, 1975)
- `HC3`
    - A modification of HC0 that approximates a jackknife estimator. Squared residuals are divided by the square of 1-h (Davidson and Mackinnon, 1993).
- `HC4`
    - A modification of HC0 that divides the squared residuals by 1-h to a power that varies according to h, n, and p, with an upper limit of 4 (Cribari-Neto, 2004).
- `HC4m`
    - Similar to HC4 but with smaller bias (Cribari-Neto and Da Silva, 2011)
- `HC5`
    - A modification of HC0 that divides the squared residuals by 1-h to a power that varies according to h, n, and p, with an upper limit of 4 (Cribari-Neto, 2004). (Cribari-Neto, Souza and Vasconcellos, 2007)


To get a feel of how the use of different estimators impact inference, we conduct a simple Monte Carlo:
```
using CovarianceMatrices
using GLM

function montecarlo()
    simulations = 1000
    nobs = 50
    p = 5
    gamma = [0.1 for j in 1:p]

    results = Array{NTuple{7,Array{Float64,1}}, 1}()


for j in 1:simulations
    ## Simulate y = X*beta_0 + exp(X*gamma)*u; beta_0 = [0,...,0], u ~ N(0,1), X ~ N(0,I_p)
    @show j
    X = randn(nobs, p)
    u = randn(nobs)
    y = randn(nobs) .+ exp.(X*gamma).*u

    OLS = fit(LinearModel, X, y)

    v = (stderror(OLS, HC0())),
        (stderror(OLS, HC1())),
        (stderror(OLS, HC2())),
        (stderror(OLS, HC3())),
        (stderror(OLS, HC4())),
        (stderror(OLS, HC4m())),
        (stderror(OLS, HC5()))

    push!(results, v)
end

return v
end

# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
# The weights are added just to test the interface and are not part
# of the original data
clotting = DataFrame(
    u    = log.([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = 9*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)
wOLS = fit(GeneralizedLinearModel, @formula(lot1~u), clotting, Normal(), wts = convert(Array{Float64}, clotting[:w]))

vcov(wOLS, HC0)
vcov(wOLS, HC1)
vcov(wOLS, HC2)
vcov(wOLS, HC3)
vcov(wOLS, HC4)
vcov(wOLS, HC4m)
vcov(wOLS, HC5)
```

## CRHC (Cluster robust heteroskedasticty consistent)
The API of this class of variance estimators is subject to change, so please use with care. The difficulty is that `CRHC` type needs to have access to the variable along which dimension the clustering mast take place. For the moment, the following approach works --- as long as no missing values are present in the original dataframe.

```julia
using RDatasets
df = dataset("plm", "Grunfeld")
lm = glm(@formula(Inv~Value+Capital), df, Normal(), IdentityLink())
vcov(lm, CRHC1(convert(Array{Float64}, df[:Firm])))
stderror(lm, CRHC1(convert(Array{Float64}, df[:Firm])))
```
