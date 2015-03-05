# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=devel)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)
[![Coverage Status](https://img.shields.io/coveralls/gragusa/CovarianceMatrices.jl.svg)](https://coveralls.io/r/gragusa/CovarianceMatrices.jl?branch=devel)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Status

- [x] Basic interface
- [x] Implement Type0-Type4 (HC0, HC1, HC2, HC3, HC4) variances
- [x] Implement Type4m-Type5
- [x] HAC with manual bandwidth
- [ ] HAC automatic bandwidth (Andrews)
  - [x] AR(1) model
  - [ ] AR(p) model
  - [ ] MA(p) model
  - [ ] ARMA(p) model
- [ ] HAC automatic bandwidth (Newey-West)
- [ ] Extend API to allow passing option to automatic bandwidth selection methods
- [x] Cluster Robust HC
- [ ] Two-way cluster robust
- [x] Compatible with  `GLM.jl`
  - [x] HC
  - [x] HAC
  - [x] Cluster Robust


## Install

```
Pkg.add("CovarianceMatrices.jl")
```

## Basic usage

### Heteroskedasticity Robust

```
using CovarianceMatrices
using DataFrames
using GLM

# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
# The weights are added just to try the full interface.
clotting = DataFrame(
    u    = log([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = [1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)
wOLS = fit(GeneralizedLinearModel, lot1~u, clotting, Normal(), wts = array(clotting[:w]))

vcov(wOLS, HC0())
vcov(wOLS, HC1())
vcov(wOLS, HC2())
vcov(wOLS, HC3())
vcov(wOLS, HC4())

stderr(wOLS, HC0())
stderr(wOLS, HC1())
stderr(wOLS, HC2())
stderr(wOLS, HC3())
stderr(wOLS, HC4())

```

### Heteroskedasticity and Autocorrelation Robust

```
## Simulated AR(1) and estimate it using OLS
srand(1)
y = zeros(Float64, 100)
rho = 0.8
y[1] = randn()
for j = 2:100
  y[j] = rho * y[j-1] + randn()
end

data = DataFrame(y = y[2:100], yl = y[1:99])
AR1  = fit(GeneralizedLinearModel, y~yl, data, Normal())

## Default assume iid (which is correct in this case)
vcov(AR1)

## The truncated kernel (TruncatedKernel)
vcov(AR1, TruncatedKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, TruncatedKernel(1))
vcov(AR1, TruncatedKernel(2))
vcov(AR1, TruncatedKernel())  ## Optimal bandwidth

## The Bartelett kernel
vcov(AR1, BartlettKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, BartlettKernel(1))
vcov(AR1, BartlettKernel(2))
vcov(AR1, BartlettKernel())  ## Optimal bandwidth
vcov(AR1, BartlettKernel(), prewhite = true)

## The Parzent kernel
vcov(AR1, ParzenKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, ParzenKernel(1))
vcov(AR1, ParzenKernel(2))
vcov(AR1, ParzenKernel())  ## Optimal bandwidth

## The quadratic-spectral kernel
vcov(AR1, QuadraticSpectralKernel(0.1)) ## Same as iid because bandwidth = 0
vcov(AR1, QuadraticSpectralKernel(.5))
vcov(AR1, QuadraticSpectralKernel(2.))
vcov(AR1, QuadraticSpectralKernel())  ## Optimal bandwidth

```
