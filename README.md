# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=master)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)
[![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=master)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Introduction

This package provides types and methods useful to obtain consistent estimates of the long run covariance matrix of a random process.

Three classes of estimators are considered:

1. **HAC** 
    - heteroskedasticity and autocorrelation consistent (Andrews, 1996; Newey and West, 1994)
2. **HC**  
    - heteroskedasticity consistent (White, 1982)
3. **CRVE** 
    - cluster robust (Arellano, 1986; Bell, 2002)

The typical applications of these estimators are __(a)__ estimation of the long run covariance matrix of a stochastic process, __(b)__ robust inference about parameters of regular statistical models such as LM and GLM.


## Installation

The package is registered on [METADATA](http::/github.com/JuliaLang/METADATA.jl), so to install
```julia
pkg> add CovarianceMatrices
```

## Quick and dirty

In this section we will describe the typical application of this package, that is, obtaining a consistent estimate of the covariance of coefficient of generalized linear model. 

To get the covariance matrix of the estimated coefficients of a generalized linear model fitted through `GLM` use `vcov`

```julia
vcov(::DataFrameRegressionModel, ::T) where T<:RobustVariance
```
Standard errors can be obtained by 
```julia
stderror(::DataFrameRegressionModel, ::T) where T<:RobustVariance
```

`RobustVariance` is an abstract type. Subtypes of `RobustVariance` are `HAC`, `HC` and `CRHC`. These in turns are themselves abstract with the following concrete types:

- `HAC`
    - `TruncatedKernel`
    - `BartlettKernel`
    - `ParzenKernel`
    - `TukeyHanningKernel`
    - `QuadraticSpectralKernel `

- `HC`
    - `HC1`
    - `HC2`
    - `HC3`
    - `HC4`
    - `HC4m`
    - `HC5`

- `CRHC`
    - `CRHC1`
    - `CRHC2`
    - `CRHC3`


### HAC (Heteroskedasticity and Autocorrelation Consistent)

```julia
vcov(::DataFrameRegressionModel, ::T) where T<:HAC
```

Consider the following artificial data (a regression with autoregressive error component):
```julia
using CovarianceMatrices
using DataFrames
using Random
Random.seed!(1)
n = 500
x = randn(n,5)
u = Array{Float64}(undef, n)
u[1] = rand()
for j in 2:n
    u[j] = 0.78*u[j-1] + randn()
end
df = DataFrame()
df[:y] = u
for j in enumerate([:x1, :x2, :x3, :x4, :x5])
    df[j[2]] = x[:,j[1]]
end
```

Using the data in `df`, the coefficient of the regression can be estimated using `GLM`

```julia
lm1 = lm(@formula(y~x1+x2+x3+x4+x5), df)
```

Given the autocorrelation structure, a consistent covariance of the coefficient requires to use a `HAC` estimator. For instance, the follwoing
```julia
vcov(lm1, ParzenKernel(1.34, prewhiten = true))
```
will give an estimate based on the Parzen Kernel with bandwidth 1.34. The kernel option `prewhitening = true` indicates that the covariance will calculated after prewhitening (the dafault is `prewhiten=false`). 

Typically, the bandwidth is chosen by data-dependent procedures. These procedures are given in Andrews (1996) and Newey and West (1994). 

To get an estimate with bandwidth selected automatically following Andrews, 
```julia
vcov(lm1, ParzenKernel(Andrews))
```
This works for all kernels, e.g.,
```julia
vcov(lm1, TruncatedKernel(Andrews))
vcov(lm1, BartlettKernel(Andrews))
vcov(lm1, QuadraticSpectralKernel(Andrews))
```

To use instead Newey West approach, 
```julia
vcov(lm1, QuadraticSpectralKernel(NeweyWest))
```
Newey West selection only works for the `QuadraticSpectralKernel`, `BartlettKernel`, and `ParzenKernel`. 

To retrive the optimal bandwidth calculated by these two procedures, the kernel must be preallocated. For instance,
```julia
kernel = QuadraticSpectralKernel(NeweyWest)
vcov(lm1, kernel)
```
Now `kernel.bw` containes the optimal bandwidth. 


### HC (Heteroskedastic consistent)

As in the HAC case, `vcov` and `stderr` are the main functions. They know get as argument the type of robust variance being sought
```julia
vcov(::DataFrameRegressionModel, ::HC)
```
Where `HC` is an abstract type with the following concrete types:

- `HC0`
    - This is Hal White (1982) robust variance estimator
- `HC1`
    - This is equal to `H0` multiplyed it by _n/(n-p)_, where n is the sample size and _p_ is the number of parameters in the model.
- `HC2`
    - A modification of `HC0` that involves dividing the squared residual by _1-h_, where h is the leverage for the case (Horn, Horn and Duncan, 1975)
- `HC3`
    - A modification of `HC0` that approximates a jackknife estimator. Squared residuals are divided by the square of _1-h_ (Davidson and Mackinnon, 1993).
- `HC4`
    - A modification of `HC0` that divides the squared residuals by _1-h_ to a power that varies according to _h_, _n_, and _p_. (Cribari-Neto, 2004).
- `HC4m`
    - Similar to `HC4` but with smaller bias (Cribari-Neto and Da Silva, 2011)
- `HC5`
    - A modification of `HC0` that divides the squared residuals by the square of _1-h_ to a power that varies according to _h_, _n_, and _p_. (Cribari-Neto, 2004). (Cribari-Neto, Souza and Vasconcellos, 2007)

Example:

```julia
obs = 50
df = DataFrame(x = randn(obs))
df[:y] = df[:x] + sqrt.(df[:x].^2).*randn(obs)
lm = fit(LinearModel,@formula(y~x), df)
vcov(ols)
vcov(ols, HC0())
vcov(ols, HC1())
vcov(ols, HC2())
vcov(ols, HC3())
vcov(ols, HC4m())
vcov(ols, HC5())
```


### CRHC (Cluster robust heteroskedasticty consistent)
The API of this class of variance estimators is subject to change, so please use with care. The difficulty is that `CRHC` type needs to have access to the variable along which dimension the clustering mast take place. For the moment, the following approach works --- as long as no missing values are present in the original dataframe.

Example:

```julia
using RDatasets
df = dataset("plm", "Grunfeld")
lm = glm(@formula(Inv~Value+Capital), df, Normal(), IdentityLink())
```

The `Firm` clustered variance and standard errors can be obtained: 

```julia
vcov(lm, CRHC1(convert(Array{Float64}, df[:Firm])))
stderror(lm, CRHC1(convert(Array{Float64}, df[:Firm])))
```
