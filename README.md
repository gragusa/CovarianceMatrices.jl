# CovarianceMatrices.jl

[![Build Status](https://travis-ci.org/gragusa/CovarianceMatrices.jl.svg?branch=v0.4)](https://travis-ci.org/gragusa/CovarianceMatrices.jl)

[![CovarianceMatrices](http://pkg.julialang.org/badges/CovarianceMatrices_0.4.svg)](http://pkg.julialang.org/?pkg=CovarianceMatrices&ver=0.4)

[![Coverage Status](https://coveralls.io/repos/gragusa/CovarianceMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CovarianceMatrices.jl?branch=v0.4)
[![codecov.io](http://codecov.io/github/gragusa/CovarianceMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CovarianceMatrices.jl?branch=v0.4)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Status

- [x] Basic interface
- [x] Implement Type0-Type4 (HC0, HC1, HC2, HC3, HC4) variances
- [x] Implement Type5-Type5c
- [x] HAC with manual bandwidth
- [x] HAC automatic bandwidth (Andrews)
  - [x] AR(1) model
  - [ ] AR(p) model
  - [ ] MA(p) model
  - [ ] ARMA(p) model

- [ ] HAC automatic bandwidth (Newey-West)
- [ ] De-meaned versions: E[(X-\mu)(X-\mu)']
- [x] VARHAC
- [ ] Extend API to allow passing option to automatic bandwidth selection methods
- [x] Cluster Robust HC
- [ ] Two-way cluster robust
- [x] Interface with `GLM.jl`
- [ ] Drop-in `show` method for `GLM.jl`

## Install

```
Pkg.add("CovarianceMatrices")
```

## Basic usage

### Heteroskedasticity Robust

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

## Truncated kernel (TruncatedKernel)
vcov(AR1, TruncatedKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, TruncatedKernel(1))
vcov(AR1, TruncatedKernel(2))
vcov(AR1, TruncatedKernel())  ## Optimal bandwidth

## Bartelett kernel
vcov(AR1, BartlettKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, BartlettKernel(1))
vcov(AR1, BartlettKernel(2))
vcov(AR1, BartlettKernel())  ## Optimal bandwidth

## Parzen kernel
vcov(AR1, ParzenKernel(0)) ## Same as iid because bandwidth = 0
vcov(AR1, ParzenKernel(1))
vcov(AR1, ParzenKernel(2))
vcov(AR1, ParzenKernel())  ## Optimal bandwidth

## Quadratic-Spectral kernel
vcov(AR1, QuadraticSpectralKernel(0.1))
vcov(AR1, QuadraticSpectralKernel(.5))
vcov(AR1, QuadraticSpectralKernel(2.))
vcov(AR1, QuadraticSpectralKernel())  ## Optimal bandwidth

```
