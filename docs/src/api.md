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

## HAC Estimators

### Kernel Types

```@docs
Bartlett
Parzen
QuadraticSpectral
Truncated
TukeyHanning
```

### Bandwidth Selection Types

```@docs
Andrews
NeweyWest
```

## Heteroskedasticity-Robust Estimators

```@docs
HC0
HC1
HC2
HC3
HC4
HC5
```

### Alternative Names

```@docs
HR0
HR1
HR2
HR3
HR4
HR5
```

## Clustered Standard Errors

```@docs
CR0
CR1
CR2
CR3
```

### Cached Cluster Estimators

For high-performance repeated calculations (e.g., wild bootstrap):

```@docs
CachedCR
CRCache
```

## Advanced Estimators

### VARHAC

```@docs
VARHAC
AICSelector
BICSelector
FixedSelector
SameLags
FixedLags
AutoLags
DifferentOwnLags
```

### Smoothed Moments

```@docs
SmoothedMoments
UniformSmoother
TriangularSmoother
```

### Panel Data Estimators

```@docs
DriscollKraay
```

### Non-parametric Estimators

```@docs
EWC
```

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
score
objective_hessian
weight_matrix
```

## Legacy Support

### Deprecated Smoothers

```@docs
BartlettSmoother
TruncatedSmoother
```

Note: These are deprecated in favor of `SmoothedMoments` with appropriate kernels.

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
- `smooth_moments_threaded!`: Multi-threaded smoothing
- `compute_weights`: Compute smoothing weights (fallback)
- `compute_normalization`: Normalization constants

### Internal Utility Functions

- `finalize_prewhite`: Handle prewhitening operations
- `fit_var`: VAR model fitting for prewhitening/VARHAC
- `rdiv!`: In-place division
- `groupby`: Grouping operations for clustered data

These internal functions are subject to change and should generally not be used directly in user code.