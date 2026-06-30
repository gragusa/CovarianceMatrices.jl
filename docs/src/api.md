# API Reference

```@meta
CurrentModule = CovarianceMatrices
```

This section provides comprehensive documentation for all exported functions and types in CovarianceMatrices.jl.

## Main Functions

### Core Estimation Function

```@docs
aVar
```

### GLM Integration

```@docs
StatsBase.vcov
StatsBase.stderror
```

### Bandwidth Selection

```@docs
optimalbw
```

## Abstract Types

```@docs
AbstractAsymptoticVarianceEstimator
Uncorrelated
Correlated
```

## HAC Estimator Reference

### Kernel Type Reference

The kernel types are documented on the [HAC Estimators](estimators/hac.md) page:
[`Bartlett`](@ref), [`Parzen`](@ref), [`QuadraticSpectral`](@ref),
[`Truncated`](@ref), and [`TukeyHanning`](@ref).

### Bandwidth Selection Type Reference

The data-driven bandwidth selectors [`Andrews`](@ref) and [`NeweyWest`](@ref) are
documented on the [HAC Estimators](estimators/hac.md) page.

```@docs
Fixed
```

## Heteroskedasticity-Robust Reference

The HC estimators [`HC0`](@ref), [`HC1`](@ref), [`HC2`](@ref), [`HC3`](@ref),
[`HC4`](@ref), and [`HC5`](@ref), and their `HR` aliases [`HR0`](@ref),
[`HR1`](@ref), [`HR2`](@ref), [`HR3`](@ref), [`HR4`](@ref), and [`HR5`](@ref),
are documented on the [Heteroskedasticity-Robust (HC/HR) Estimators](@ref) page.

```@docs
HC4m
HR4m
```

## Clustered Standard Errors Reference

The cluster-robust estimators [`CR0`](@ref), [`CR1`](@ref), [`CR2`](@ref), and
[`CR3`](@ref), and the cached estimator [`CachedCR`](@ref), are documented on the
[Clustered Robust (CR) Estimators](@ref) page.

```@docs
CachedCRModel
CRCache
CRModelCache
```

## Advanced Estimator Reference

### VARHAC

[`VARHAC`](@ref) and its lag selectors and strategies ([`AICSelector`](@ref),
[`BICSelector`](@ref), [`FixedSelector`](@ref), [`SameLags`](@ref),
[`FixedLags`](@ref), [`AutoLags`](@ref), [`DifferentOwnLags`](@ref)) are
documented on the [VARHAC: Vector Autoregression HAC Estimation](@ref) page.

### Smoothed Moments

[`UniformSmoother`](@ref) and [`TriangularSmoother`](@ref) are documented on the
[Smoothed Moments Estimation](estimators/smoothed_moments.md) page.

### Panel Data Estimators

[`DriscollKraay`](@ref) is documented on the [Driscoll-Kraay Estimator](@ref) page.

### Non-parametric Estimators

[`EWC`](@ref) is documented on the [Equal Weighted Cosine (EWC) Estimator](@ref) page.

## Utility Functions

### VARHAC Utilities

```@docs
AICs
BICs
order_aic
order_bic
order
maxlags
```

### HAC Utilities

```@docs
bandwidth
```

## New Unified API

### Variance Forms

```@docs
Information
Misspecified
VarianceForm
```

### Model Types

```@docs
MLikeModel
GMMLikeModel
```

### Model Interface Functions

```@docs
momentmatrix
cross_score
hessian_objective
jacobian_momentfunction
weight_matrix
```

### Variance Specification Wrapper

```@docs
VcovSpec
```

## Internal Types and Functions

The following functions are primarily for internal use but may be useful for advanced users:

### Internal Computation Functions

- `avar`: Low-level covariance computation (method-specific)
- `setkernelweights!`: Set kernel weights for HAC estimators
- `workingoptimalbw`: Internal bandwidth computation
- `scalevar!`: Scale variance matrices
- `demeaner`: Remove means from data matrices

### Internal HAC Functions

- `_optimalbandwidth`: Compute optimal bandwidth
- `kernelfunction`: Evaluate kernel functions
- `avar_func`: Core HAC computation

### Internal Smoothing Functions

- `smooth_moments!`: In-place moment smoothing (kernel-based)
- `compute_weights`: Compute smoothing weights (fallback)
- `compute_normalization`: Normalization constants

### Internal Utility Functions

- `finalize_prewhite`: Handle prewhitening operations
- `fit_var`: VAR model fitting for prewhitening/VARHAC
- `rdiv!`: In-place division
- `groupby`: Grouping operations for clustered data

These internal functions are subject to change and should generally not be used directly in user code.