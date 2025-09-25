# CovarianceMatrices.jl

*A Julia package for robust covariance matrix estimation*

```@meta
CurrentModule = CovarianceMatrices
```

## Overview

CovarianceMatrices.jl provides a comprehensive suite of robust covariance matrix estimators for econometric applications. The package handles various forms of dependence in data including:

- **Heteroskedasticity** (HC/HR estimators)
- **Serial correlation** (HAC estimators)
- **Cluster correlation** (CR estimators)
- **Cross-sectional and temporal dependence** (Driscoll-Kraay)
- **Complex dependence structures** (VARHAC, Smoothed Moments, EWC)

## Key Features

- **🎯 Unified API**: Single interface for all estimators with both matrix and GLM.jl integration
- **📊 Comprehensive Coverage**: Full range of robust covariance estimators used in econometrics
- **⚡ High Performance**: Optimized implementations with threading support where applicable
- **🔬 Mathematical Rigor**: Implementations based on latest econometric theory and best practices
- **📖 Rich Documentation**: Detailed mathematical explanations and practical tutorials

## Quick Start

### Basic Matrix Interface

```julia
using CovarianceMatrices

# Generate some time series data with serial correlation
T = 1000
X = cumsum(randn(T, 3), dims=1)  # Random walk data

# HAC estimation with Newey-West
hac_est = Bartlett{NeweyWest}()
Ω_hac = aVar(hac_est, X)

# VARHAC estimation (no bandwidth selection needed)
varhac_est = VARHAC()
Ω_varhac = aVar(varhac_est, X)

# Smoothed moments (automatic positive semi-definiteness)
smooth_est = SmoothedMoments()
Ω_smooth = aVar(smooth_est, X)
```

### GLM Integration

```julia
using GLM, CovarianceMatrices

# Fit regression model
data = (y = randn(100), x1 = randn(100), x2 = randn(100))
model = lm(@formula(y ~ x1 + x2), data)

# Robust standard errors
vcov_hc = vcov(HC3(), model)
se_hc = stderror(HC3(), model)

# HAC standard errors
vcov_hac = vcov(Parzen{Andrews}(), model)
se_hac = stderror(Parzen{Andrews}(), model)
```

## Package Structure

```@docs
AVarEstimator
```

The package is organized around the `AVarEstimator` abstract type, with specific estimators for different types of dependence:

### Heteroskedasticity-Robust Estimators
- [`HC0`](@ref), [`HC1`](@ref), [`HC2`](@ref), [`HC3`](@ref), [`HC4`](@ref), [`HC5`](@ref)
- Same functionality available as [`HR0`](@ref)-[`HR5`](@ref)

### HAC (Heteroskedasticity and Autocorrelation Consistent) Estimators
- [`Bartlett`](@ref), [`Parzen`](@ref), [`QuadraticSpectral`](@ref)
- [`Truncated`](@ref), [`TukeyHanning`](@ref)
- Bandwidth selection: [`Andrews`](@ref), [`NeweyWest`](@ref), or fixed

### Clustered Standard Errors
- [`CR0`](@ref), [`CR1`](@ref), [`CR2`](@ref), [`CR3`](@ref)

### Advanced Estimators
- [`VARHAC`](@ref): VAR-based HAC estimation without bandwidth selection
- [`SmoothedMoments`](@ref): Smith's smoothed moments with automatic PSD
- [`DriscollKraay`](@ref): Panel data with spatial and temporal correlation
- [`EWC`](@ref): Equal Weighted Cosine estimation

## Main Functions

```@docs
aVar
vcov
stderror
```

## Installation

```julia
using Pkg
Pkg.add("CovarianceMatrices")
```

## Citation

If you use CovarianceMatrices.jl in your research, please cite:

```bibtex
@misc{CovarianceMatricesJl,
  title = {CovarianceMatrices.jl: Robust Covariance Matrix Estimation for Julia},
  author = {Giuseppe Ragusa and contributors},
  year = {2024},
  url = {https://github.com/gragusa/CovarianceMatrices.jl}
}
```