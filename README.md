# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=master)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)
[![CovarianceMatrices](http://pkg.julialang.org/badges/CovarianceMatrices_0.5.svg)](http://pkg.julialang.org/?pkg=CovarianceMatrices&ver=0.5)
[![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=master)
[![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=master)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Installation

The package is registered on [METADATA](http::/github.com/JuliaLang/METADATA.jl), so installing it amounts to issue
```julia
Pkg.add("CovarianceMatrices")
```

## Introduction

This package provides types and methods useful to obtain consistent estimates of the long run covariance matrix of a random process.

Three classes of estimators are considered:

1. **HAC** - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West)
2. **HC**  - hetheroskedasticity (White, 1982)
3. **CRVE** - cluster robust (Arellano, 1986)

The typical application of these estimators is in inference about the parameters of a models. Accordingly, this package extends methods defined in [StatsBase.jl](http://github.com/JuliaStat/StatsBase.jl) and [GLM.jl](http://github.com/JuliaStat/GLM.jl) to make it easy obtaining inference that is robust to heteroskedasticity and/or autocorrelation, or presence of cluster components.

# Quick tour

## HAC (Heteroskedasticity and Autocorrelation Consistent)

Available kernel types are:

- `TruncatedKernel`
- `BartlettKernel`
- `ParzenKernel`
- `TukeyHanningKernel`
- `QuadraticSpectralKernel`

For example, `ParzenKernel(NeweyWest)` return an instance of `TruncatedKernel` parametrized by `NeweyWest`, the type that corresponds to the optimal bandwidth calculated following Newey and West (1994).  Similarly, `ParzenKernel(Andrews)` corresponds to the optimal bandwidth obtained in Andrews (1991). If the bandwidth is known, it can be directly passed, i.e. `TruncatedKernel(2)`.

The examples below hopefully clarify the API, that is however relatively simple to use.

### Long run variance of the regression coefficient

In the regression context, the function `vcov` does all the work. Its API` is
```julia
vcov(::DataFrameRegressionModel, ::HAC; prewhite = true)
```

The example below, describe a typical call to it on a regression model on generated data (a regression with autoregressive error component):
```julia
using CovarianceMatrices
using DataFrames
srand(1)
n = 500
x = randn(n,5)
u = Array{Float64}(2*n)
u[1] = rand()
for j in 2:2*n
    u[j] = 0.78*u[j-1] + randn()
end
u = u[n+1:2*n]    
y = 0.1 + x*[0.2, 0.3, 0.0, 0.0, 0.5] + u            

df = DataFrame()
df[:y] = y
for j in enumerate([:x1, :x2, :x3, :x4, :x5])
    df[j[2]] = x[:,j[1]]
end
```
Using the data in `df`, the coefficient of the regression can be estimated using `GLM`

```julia
lm1 = glm(y~x1+x2+x3+x4+x5, df, Normal(), IdentityLink())
```

To get a consistent estimate of the long run variance of the estimated coefficient using a Quadratic Spectral kernel with automatic bandwidth selection a la' Andrews
```julia
vcov(lm1, QuadraticSpectralKernel(Andrews), prewhite = false)
```
If one wants to estimate the long-time variance using the same kernel, but with a bandwidth selected a la' Newey-West
```julia
vcov(lm1, QuadraticSpectralKernel(NeweyWest), prewhite = false)
```
The standard errors can be obtained by the `stderr` function
```julia
vcov(::DataFrameRegressionModel, ::HAC; prewhite = true)
```
Sometime is useful to access the bandwidth automatically selected. This can be done using the `optimalbw`
```julia
optimalbw(NeweyWest, QuadraticSpectralKernel, lm1; prewhite = false)
optimalbw(Andrews, QuadraticSpectralKernel, lm1; prewhite = false)
```

### Long run variance of the average of the process

Sometime is of interest estimating the long-run variance of the average of the process. At the moment this can be done by carrying out a regression on a constant (the sample mean of the realization of the process) and using `vcov` or `stderr` to obtain a consistent variance.

```julia
lm2 = glm(u~1, df, Normal(), IdentityLink())
vcov(lm1, ParzenKernel(NeweyWest), prewhite = false)
stderr(lm1, ParzenKernel(NeweyWest), prewhite = false)
```

## HC (Heteroskedastic consistent)

As in the HAC case, `vcov` and `stderr` are the main functions. They know get as argument the type of robust variance being sought
```julia
vcov(::DataFrameRegressionModel, ::HC)
```
Where HC is an abstract type with the following concrete types:

- `HC0`
- `HC1` (this is `HC0` with the degree of freedom adjustment)
- `HC2`
- `HC3`
- `HC4`
- `HC4m`
- `HC5`


```
using CovarianceMatrices
using DataFrames
using GLM

# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
# The weights are added just to test the interface and are not part
# of the original data
clotting = DataFrame(
    u    = log([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = 9*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)
wOLS = fit(GeneralizedLinearModel, lot1~u, clotting, Normal(), wts = array(clotting[:w]))

vcov(wOLS, HC0
vcov(wOLS, HC1)
vcov(wOLS, HC2)
vcov(wOLS, HC3)
vcov(wOLS, HC4)
vcov(wOLS, HC4m)
vcov(wOLS, HC5)
```

## CRHC (Cluster robust heteroskedasticty consistent)
The API of this class of variance estimators is subject to change, so please use with care. In particular, since the `CRHC` type needs to know with variable is used for clustering---a good way pass it must be thought out. For the moment, the following approach works --- as long as no missing values are present in the original dataframe.

```julia
using RDatasets
df = dataset("plm", "Grunfeld")
lm = glm(Inv~Value+Capital, df, Normal(), IdentityLink())
vcov(lm, CRHC1(convert(Array, df[:Firm])))
stderr(lm, CRHC1(convert(Array, df[:Firm])))
```
