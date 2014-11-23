# VCOV.jl

[![Build Status](https://travis-ci.org/gragusa/VCOV.jl.svg?branch=master)](https://travis-ci.org/gragusa/VCOV.jl)

Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.

## Install

```
Pkg.clone("VCOV.jl")
```

## Usage

```
using VCOV
X = randn(100,4)
## Quadratic Spectral Kernel with band-width 2
vcov(X, QuadraticSpectral(2))
## Quadratic Spectral Kernel with automathic band-width 
vcov(X, QuadraticSpectral())
```
