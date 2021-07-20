# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=master)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)
[![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=master)
[![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=master)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Installation


```julia
Pkg.add("CovarianceMatrices")
```

---

## Introduction

This package provides types and methods useful to obtain consistent estimates of the long run covariance matrix of a random process.

Three classes of estimators are considered:

1. **HAC** - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West, 1994)
2. **VARHAC** - Vector Autoregression based HAC (Den Haan and Levine)
3. **Smoothed** - (Smith, 2014)
2. **HC**  - hetheroskedasticity consistent (White, 1982)
3. **CRVE** - cluster robust (Arellano, 1986)

The typical application of these estimators is to conduct robust inference about parameters of a statistical model. 

The package extends methods defined in [StatsBase.jl](http://github.com/JuliaStat/StatsBase.jl) and [GLM.jl](http://github.com/JuliaStat/GLM.jl) to provide a plug-and-play replacement for the  standard errors calculated by default by [GLM.jl](http://github.com/JuliaStat/GLM.jl).

The API can be used regardless of whether the model is fit with [GLM.jl](http://github.com/JuliaStat/GLM.jl) and developer can extend their fit functions to provides robust standard errors. 

# Quick tour

## HAC (Heteroskedasticity and Autocorrelation Consistent)

Available kernel types are:

- `TruncatedKernel`
- `BartlettKernel`
- `ParzenKernel`
- `TukeyHanningKernel`
- `QuadraticSpectralKernel`

For example, `ParzenKernel{NeweyWest}()` return an instance of `TruncatedKernel` parametrized by `NeweyWest`, the type that corresponds to the optimal bandwidth calculated following Newey and West (1994).  Similarly, `ParzenKernel{Andrews}()` corresponds to the optimal bandwidth obtained in Andrews (1991). If the bandwidth is known, it can be directly passed, i.e. `TruncatedKernel(2)`.


### Long run variance of the regression coefficient

In the regression context, the function `vcov` does all the work:
```julia
vcov(::HAC, ::DataFrameRegressionModel; prewhite = true)
```

Consider the following artificial data (a regression with autoregressive error component):

```julia
using CovarianceMatrices
using Random, DataFrames, GLM
Random.seed!(1)
n = 500
x = randn(n,5)
u = zeros(2*n)
u[1] = rand()
for j in 2:2*n
    u[j] = 0.78*u[j-1] + randn()
end
u = u[n+1:2*n]
y = 0.1 .+ x*[0.2, 0.3, 0.0, 0.0, 0.5] + u

df = convert(DataFrame,x)
df[!,:y] = y
```
Using the data in `df`, the coefficient of the regression can be estimated using `GLM`

```julia
lm1 = glm(@formula(y~x1+x2+x3+x4+x5), df, Normal(), IdentityLink())
```

To get a consistent estimate of the long run variance of the estimated coefficients using a Quadratic Spectral kernel with automatic bandwidth selection  _à la_ Andrews
```julia
vcov(QuadraticSpectralKernel{Andrews}(), lm1, prewhite = false)
```
If one wants to estimate the long-time variance using the same kernel, but with a bandwidth selected _à la_ Newey-West
```julia
vcov(QuadraticSpectralKernel{NeweyWest}(), lm1, prewhite = false)
```
The standard errors can be obtained by the `stderror` method
```julia
stderror( ::HAC, ::DataFrameRegressionModel; prewhite::Bool)
```
For the previous example:
```julia
stderror(QuadraticSpectralKernel{NeweyWest}(), lm1, prewhite = false)
```

Sometime is useful to access the bandwidth selected by the automatic procedures. This can be done using the `optimalbandwidth` method
```julia
optimalbandwidth(QuadraticSpectralKernel{NeweyWest}(), lm1; prewhite = false)
optimalbandwidth(QuadraticSpectralKernel{Andrews}(), lm1; prewhite = false)
```
Alternatively, the optimal bandwidth is stored in the kernel structure (upon calculation of the variance) and can be accessed. This requires however that the kernel type is materialized:
```julia
kernel = QuadraticSpectralKernel{NeweyWest}()
stderror(kernel, lm1, prewhite = false)
bw = CovarianceMatrices.bandwidth(kernel)
```


### Covariances without `GLM.jl`

One might want to calculate variance estimator when the regression (or some other model) is fit "manually". Below is an example of how this can be accomplished.

```julia
X   = [ones(n) x]
_,K = size(X)
b   = X\y
res = y .- X*b
momentmatrix = X.*res
B   = inv(X'X)                                                         # Jacobian of moment conditions
A   = lrvar(QuadraticSpectralKernel(bw[1]), momentmatrix, scale = n^2/(n-K))   # df adjustment is built into vcov
Σ   = B*A*B
Σ .- vcov(QuadraticSpectralKernel(bw[1]), lm1, dof_adjustment=true)
```
The utility function `sandwich` does all this automatically:

```julia
vcov(QuadraticSpectralKernel(bw[1]), lm1, dof_adjustment=true) ≈ CovarianceMatrices.sandwich(QuadraticSpectralKernel(bw[1]), B, momentmatrix, dof = K)
vcov(QuadraticSpectralKernel(bw[1]), lm1, dof_adjustment=false) ≈ CovarianceMatrices.sandwich(QuadraticSpectralKernel(bw[1]), B, momentmatrix, dof = 0)
```


## HC (Heteroskedastic consistent)

As in the HAC case, `vcov` and `stderror` are the main functions. They know get as argument the type of robust variance being sought

```julia
vcov(::HC, ::DataFrameRegressionModel)
```

Where HC is an abstract type with the following concrete types:

- `HC0`
- `HC1` (this is `HC0` with the degree of freedom adjustment)
- `HC2`
- `HC3`
- `HC4`
- `HC4m`
- `HC5`


```julia
using CovarianceMatrices, DataFrames, GLM
# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
# The weights are added just to test the interface and are not part
# of the original data
clotting = DataFrame(
    u    = log.([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = 9*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)
wOLS = fit(GeneralizedLinearModel, @formula(lot1~u), clotting, Normal(), wts = clotting[!,:w])

vcov(HC0(),wOLS)
vcov(HC1(),wOLS)
vcov(HC2(),wOLS)
vcov(HC3(),wOLS)
vcov(HC4(),wOLS)
vcov(HC4m(),wOLS)
vcov(HC5(),wOLS)

```


## CRHC (Cluster robust heteroskedasticity consistent)

The API of this class of estimators is subject to change, so please use with care. The difficulty is that `CRHC` type needs to have access to the variable along which dimension the clustering must take place. For the moment, the following approach works 

```julia
using RDatasets
df = dataset("plm", "Grunfeld")
lm = glm(@formula(Inv~Value+Capital), df, Normal(), IdentityLink())
vcov(CRHC1(:Firm, df), lm)
stderror(CRHC1(:Firm, df),lm)
```

Alternatively, the cluster indicator can be passed directly (but this will only work if there are not missing values)

```julia
vcov(CRHC1(df[:Firm]), lm)
stderror(CRHC1(df[:Firm]),lm)
```

As in the `HAC` case, `sandwich` and `lrvar` can be leveraged to constract cluster-robust variances without relying on `GLM.jl`.

## Performances


```julia
using BenchmarkTools
## Calculating a HAC on a large matrix
Z = randn(10000, 10)
@btime lrvar(BartlettKernel{Andrews}(), Z, prewhite = true) 
## 2.085 ms (180 allocations: 6.20 MiB)
```

```r
library(sandwich)
library(microbenchmark)
Z <- matrix(rnorm(10000*10), 10000, 10)
microbenchmark( "Bartlett/Newey" = {lrvar(Z, type = "Andrews", kernel = "Bartlett")})
#Unit: milliseconds
#           expr      min       lq     mean   median       uq      max     neval
# Bartlett/Andrews 135.1839 148.3426 186.1966 155.0156 246.3178 355.3174   100
```
