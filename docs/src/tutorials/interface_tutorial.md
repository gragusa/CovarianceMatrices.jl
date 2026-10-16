# Package Interface Tutorial

This tutorial demonstrates how to extend CovarianceMatrices.jl to work with custom model types. The package provides a minimal, duck-typed interface that allows any statistical model to obtain robust covariance matrices.

## Overview

CovarianceMatrices.jl uses a **duck-typing** approach: any model that implements a few required methods automatically gains access to all robust covariance estimators. You don't need to inherit from special types (though you can for semantic clarity).

### Key Benefits

1. **Minimal interface**: Implement 1-3 methods depending on your needs
2. **Automatic compatibility**: Works with all estimators (HAC, HC, CR, VARHAC, etc.)
3. **Flexible design**: Choose between Information and Misspecified variance forms
4. **Type-safe**: Optional abstract types provide compile-time guarantees

## Three Ways to Obtain a Variance

The package exposes the same estimators through three layered entry points, from
lowest to highest level:

1. **Matrix core — `aVar(estimator, Z; scale=false)`.** Takes a raw `T × m` moment
   or score matrix and returns the long-run covariance `Ω`. This is the computational
   core; every higher-level path ultimately calls it. Use it directly when you already
   hold the moment matrix and want to assemble a variance yourself (see
   [Advanced: Manual Variance Computation](#Advanced:-Manual-Variance-Computation)).

2. **Model protocol — `vcov(estimator, model)` / `vcov(estimator, form, model)` /
   `stderror(...)`.** The extension interface documented in this tutorial. A model
   that implements `momentmatrix`, `coef`, and `nobs` (plus `hessian_objective` /
   `jacobian_momentfunction` as its variance form requires) gets the full sandwich
   assembled for it. This is the surface most users implement against.

3. **Specification wrapper — `model + vcov(estimator)`.** `vcov(estimator)` with a
   single argument returns a [`VcovSpec`](@ref) that simply packages the estimator.
   The `+` operator that consumes it is defined by downstream modeling packages
   (e.g. Regress.jl), not by CovarianceMatrices.jl; this package only provides the
   wrapper so those packages can offer the `model + vcov(...)` idiom. See
   [For Modeling Packages](#For-Modeling-Packages:-Column-Named-Clusters-and-VcovSpec).

Most of this tutorial concerns surface 2, the model protocol.

## The Core Interface

The package's protocol functions (`momentmatrix`, `cross_score`, `hessian_objective`,
`jacobian_momentfunction`, `weight_matrix`) are `public` but not exported: extend them as
`CovarianceMatrices.momentmatrix(model::MyModel) = ...`, or bring them into scope with
`using CovarianceMatrices: momentmatrix`.

### Required Methods

Every model must implement:

1. **`momentmatrix(model)`**: Return the $T \times m$ matrix of moment conditions
2. **`StatsAPI.coef(model)`**: Return the $k$-dimensional parameter vector
3. **`StatsAPI.nobs(model)`**: Return the number of observations

### Optional Methods (for advanced features)

4. **`cross_score(model)`**: Return $G = \sum_i g_i g_i'$ (has default implementation)
5. **`hessian_objective(model)`**: Required for the Misspecified form (`vcov` throws if missing)
6. **`jacobian_momentfunction(model)`**: Required for `GMMLikeModel` (`vcov` throws if missing)
7. **`weight_matrix(model)`**: Optional for GMM; omit (defaults to optimal `inv(Ω)`)

## Example 1: An M-Estimator

Huber regression replaces the squared-error loss with a loss that is quadratic for
small residuals and linear beyond a cutoff `c`, which limits the influence of
outliers. The estimator solves $\sum_t \psi(y_t - x_t'\beta) x_t = 0$, so the
moment matrix is $\psi(r_t) x_t$.

```@example iface
using CovarianceMatrices, RDatasets, DataFrames, StatsAPI, StatsBase, LinearAlgebra

struct HuberRegression <: CovarianceMatrices.MLikeModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    c::Float64
end

huber_ψ(r, c) = abs(r) <= c ? r : c * sign(r)

function HuberRegression(X, y; c = 1.345, maxiter = 200, tol = 1e-10)
    β = X \ y                       # start from OLS
    for _ in 1:maxiter               # iteratively reweighted least squares
        r = y - X * β
        w = [abs(ri) <= c ? 1.0 : c / abs(ri) for ri in r]
        βnew = (X' * (w .* X)) \ (X' * (w .* y))
        converged = norm(βnew - β) < tol
        β = βnew
        converged && break
    end
    return HuberRegression(X, y, β, c)
end
nothing # hide
```

The three required methods:

```@example iface
CovarianceMatrices.momentmatrix(m::HuberRegression) =
    huber_ψ.(m.y - m.X * m.β, m.c) .* m.X

StatsAPI.coef(m::HuberRegression) = m.β
StatsAPI.nobs(m::HuberRegression) = length(m.y)
nothing # hide
```

`hessian_objective` returns the Hessian of the minimized objective, which is
positive definite at the optimum. `Information` inverts it directly, so returning
the negative log-likelihood Hessian with the wrong sign produces negative
variances.

```@example iface
function CovarianceMatrices.hessian_objective(m::HuberRegression)
    r = m.y - m.X * m.β
    w = [abs(ri) <= m.c ? 1.0 : 0.0 for ri in r]
    return m.X' * (w .* m.X)
end
nothing # hide
```

Fit the CAPM market model from the `Capm` data. Monthly returns have fat tails, so
the Huber slope differs from the OLS slope:

```@example iface
capm = dataset("Ecdat", "Capm")
y = capm.RFood .- capm.RF
X = [ones(length(y)) capm.RMRF]

model = HuberRegression(X, y)

DataFrame(coef = ["(Intercept)", "RMRF"], ols = X \ y, huber = coef(model))
```

A model carrying a variance form is asked for one explicitly. `Information` assumes
the model is correctly specified; `Misspecified` builds the sandwich:

```@example iface
DataFrame(
    coef = ["(Intercept)", "RMRF"],
    information = stderror(Information(), model),
    misspecified = stderror(HC0(), Misspecified(), model),
    misspecified_hac = stderror(Bartlett{Andrews}(), Misspecified(), model),
)
```

The sandwich standard errors are 1.5 to 2.1 times the information-form ones. Under
correct specification the two agree, so a gap this size is evidence against the
assumptions behind the information form.

## Example 2: A GMM Estimator

A `GMMLikeModel` may have more moment conditions than parameters. It implements
`jacobian_momentfunction` in place of `hessian_objective`, and optionally
`weight_matrix`.

```@example iface
struct IVGMM <: CovarianceMatrices.GMMLikeModel
    y::Vector{Float64}
    X::Matrix{Float64}   # regressors, some endogenous
    Z::Matrix{Float64}   # instruments
    β::Vector{Float64}
    W::Matrix{Float64}   # weight matrix
end

function IVGMM(y, X, Z)
    W = inv(Z' * Z)                              # two-stage least squares
    β = (X' * Z * W * Z' * X) \ (X' * Z * W * Z' * y)
    return IVGMM(y, X, Z, β, W)
end

CovarianceMatrices.momentmatrix(m::IVGMM) = (m.y - m.X * m.β) .* m.Z
CovarianceMatrices.jacobian_momentfunction(m::IVGMM) = -(m.Z' * m.X)
CovarianceMatrices.weight_matrix(m::IVGMM) = m.W
StatsAPI.coef(m::IVGMM) = m.β
StatsAPI.nobs(m::IVGMM) = length(m.y)
nothing # hide
```

The `Misspecified` form also needs the Hessian of the GMM objective,
$G'WG$ with $G$ the moment Jacobian:

```@example iface
function CovarianceMatrices.hessian_objective(m::IVGMM)
    G = CovarianceMatrices.jacobian_momentfunction(m)
    return G' * m.W * G
end
nothing # hide
```

When `weight_matrix` is supplied, `Information` computes
$(G'WG)^{-1} G'W\Omega WG (G'WG)^{-1}$, which is the same expression the
`Misspecified` form builds from this Hessian, so the two agree for this model. They
differ when the model omits `weight_matrix`: `Information` then assumes the
efficient weight $W = \Omega^{-1}$ and reduces to $(G'\Omega^{-1}G)^{-1}$.

The Grunfeld data estimate investment on firm value, instrumented by capital stock
and lagged value, giving three instruments for two parameters:

```@example iface
grunfeld = dataset("plm", "Grunfeld")
n = nrow(grunfeld)

y_iv = grunfeld.Inv
X_iv = [ones(n) grunfeld.Value]
Z_iv = [ones(n) grunfeld.Capital grunfeld.Value]

iv = IVGMM(y_iv, X_iv, Z_iv)

DataFrame(
    coef = ["(Intercept)", "Value"],
    estimate = coef(iv),
    information = stderror(HC0(), Information(), iv),
    misspecified = stderror(HC0(), Misspecified(), iv),
    clustered = stderror(CR0(grunfeld.Firm), Misspecified(), iv),
)
```

Clustering by firm widens the intercept's standard error by about 40% and narrows
the slope's by about 5%. The [GLM Integration Tutorial](glm_tutorial.md) clusters
the same data for fitted models.

## Duck Typing

Inheriting from `MLikeModel` or `GMMLikeModel` selects the variance forms available
and is the clearest way to declare which class a model belongs to. The methods
themselves are looked up by dispatch, so a type outside the hierarchy that defines
`momentmatrix`, `coef` and `nobs` works with `aVar` on its moment matrix:

```@example iface
struct QuantileRegression
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    τ::Float64
end

CovarianceMatrices.momentmatrix(m::QuantileRegression) =
    [r < 0 ? m.τ - 1 : m.τ for r in (m.y - m.X * m.β)] .* m.X
StatsAPI.coef(m::QuantileRegression) = m.β
StatsAPI.nobs(m::QuantileRegression) = length(m.y)

qr = QuantileRegression(X, y, X \ y, 0.5)
aVar(HC0(), CovarianceMatrices.momentmatrix(qr))
```

## Interface Quick Reference

### Methods Summary

| Method | Required? | Purpose | Return Type |
|--------|-----------|---------|-------------|
| `momentmatrix(model)` | ✅ Yes | Moment conditions or scores | `T × m` matrix |
| `StatsAPI.coef(model)` | ✅ Yes | Parameter estimates | `k`-vector |
| `StatsAPI.nobs(model)` | ✅ Yes | Sample size | Integer |
| `cross_score(model)` | Optional | $\sum_i g_i g_i'$ | `m × m` matrix |
| `hessian_objective(model)` | Conditional | Objective Hessian | `k × k` matrix |
| `jacobian_momentfunction(model)` | For GMM | $\partial g/\partial \beta'$ | `m × k` matrix |
| `weight_matrix(model)` | For GMM | GMM weight matrix | `m × m` matrix |

### When to Implement Each Method

**Always implement:**
- `momentmatrix(model)`: Core of the interface
- `StatsAPI.coef(model)`: Parameter vector
- `StatsAPI.nobs(model)`: Sample size

**Implement for Misspecified form:**
- `hessian_objective(model)`: Required for sandwich variance

**Implement for GMM:**
- `jacobian_momentfunction(model)`: Derivative of moments w.r.t. parameters
- Optionally `weight_matrix(model)`: If using suboptimal weight

**Usually skip (has default):**
- `cross_score(model)`: Defaults to `momentmatrix(model)' * momentmatrix(model)`

## Best Practices

### 1. Scaling Conventions

**All matrices should be unscaled** (sums, not averages):

```julia
# ✅ Correct: unscaled cross-product
function CovarianceMatrices.cross_score(model)
    g = CovarianceMatrices.momentmatrix(model)
    return g' * g  # Sum, not mean
end

# ❌ Wrong: scaled by sample size
function CovarianceMatrices.cross_score(model)
    g = CovarianceMatrices.momentmatrix(model)
    T = nobs(model)
    return (g' * g) / T  # Don't do this!
end
```

### 2. Type Stability

Ensure all methods return concretely-typed arrays:

```julia
# ✅ Good: concrete return type
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ... computation ...
    return Matrix{Float64}(result)
end

# ❌ Bad: abstract return type
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ... computation ...
    return AbstractMatrix(result)  # Type instability!
end
```

### 3. Semantic Type Hierarchy

Use abstract types when appropriate for compile-time guarantees:

```julia
# For MLE models (m = k)
struct MyMLE <: CovarianceMatrices.MLikeModel
    # ... fields ...
end

# For GMM models (m ≥ k)
struct MyGMM <: CovarianceMatrices.GMMLikeModel
    # ... fields ...
end

# For other models, no inheritance needed
struct MyCustomModel
    # ... fields ...
end
```

### 4. Documentation

Document the moment conditions clearly:

```julia
"""
    momentmatrix(model::MyModel)

Return the T × m matrix of moment conditions.

For this model, the moment conditions are:
    g_t = ψ(y_t - x_t'β) ⊗ x_t
where ψ is the Huber influence function.
"""
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ...
end
```

## Advanced: Manual Variance Computation

For maximum control, you can compute variances manually:

```julia
using CovarianceMatrices

# Your moment matrix
G = randn(500, 3)  # 500 observations, 3 moment conditions

# Estimate long-run covariance of moments
Ω_hat = aVar(VARHAC(), G; scale=false)

# If you have the Jacobian and Hessian separately
∇g = randn(3, 3)  # Jacobian of moments
H = randn(3, 3)   # Hessian of objective

# Sandwich variance (Misspecified form for MLE)
using LinearAlgebra
V_sandwich = inv(H) * Ω_hat * inv(H')

# GMM variance
V_gmm = inv(∇g' * inv(Ω_hat) * ∇g)
```

## Regression Models

A model that subtypes `StatsBase.RegressionModel` takes a separate path:
`vcov(k, model)` forms the sandwich `n · B · aVar(k, model) · B` from the model's
bread `B`, which is what the finite-sample corrections of `HC1`–`HC5` and `CR1`–`CR3`
need. The protocol is:

| Function | Required? | Returns |
|----------|-----------|---------|
| `StatsAPI.coef(model)` | Yes | `k`-vector of estimates |
| `StatsAPI.modelmatrix(model)` | Yes | `n × k` design matrix |
| [`CovarianceMatrices.bread`](@ref)`(model)` | Yes | `k × k` bread, e.g. `inv(X'X)` |
| [`CovarianceMatrices.momentresiduals`](@ref)`(model)` | Fallback: `StatsAPI.residuals` | `u` with scores `X .* u` |
| [`CovarianceMatrices.leverage`](@ref)`(model)` | `HC2`–`HC5`; fallback: `StatsAPI.leverage` | hat-matrix diagonal |
| [`CovarianceMatrices.mask`](@ref)`(model)` | Fallback: all `true` | estimable coefficients |
| `StatsAPI.weights(model)` | `CR2`/`CR3` | observation weights, empty if none |

The number of observations is the number of rows of `modelmatrix(model)`. The
fallback for `momentresiduals` is right only for unweighted least squares; a weighted
or nonlinear model defines it. An unweighted least-squares model needs little:

```@example iface
using LinearAlgebra, StatsBase

struct OLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
end
OLS(X, y) = OLS(X, y, X \ y)

StatsAPI.coef(m::OLS) = m.β
StatsAPI.modelmatrix(m::OLS) = m.X
StatsAPI.residuals(m::OLS) = m.y - m.X * m.β
StatsAPI.leverage(m::OLS) = diag(m.X * inv(m.X' * m.X) * m.X')
CovarianceMatrices.bread(m::OLS) = inv(m.X' * m.X)

ols = OLS(X, y)
DataFrame(
    coef = ["(Intercept)", "RMRF"],
    hc1 = stderror(HC1(), ols),
    hc3 = stderror(HC3(), ols),
    hac = stderror(Bartlett{Andrews}(), ols),
)
```

## For Modeling Packages: Column-Named Clusters and `VcovSpec`

A modeling package that fits models to a data table can let its users name
clusters by column and attach an estimator with `model + vcov(estimator)`. Both
features rely on the package that owns the data; CovarianceMatrices only supplies
the values.

### Column-named cluster estimators

`CR0(:firm)`, `CR1(:firm, :year)`, `Cluster(:firm)` and `Cluster((:firm, :year))`
are cluster estimators whose clusters are given by column name rather than by group
identifiers. [`CovarianceMatrices.clusters`](@ref) returns the names as a tuple of
`Symbol`s: `clusters(CR1(:firm, :year)) == (:firm, :year)`. One name means one-way
clustering, and several names mean multi-way clustering, exactly as the corresponding
tuple of vectors would.

CovarianceMatrices has no data table, so it cannot look up the names. `aVar`, `vcov`,
`stderror`, `nclusters` and `CachedCR` throw an `ArgumentError` on a
column-named estimator. The modeling package resolves it first by rebuilding the same
estimator type from the columns:

```julia
using CovarianceMatrices: CR0, CR1, CR2, CR3, Cluster, clusters
using Tables

resolve(k, data) = k
for E in (:CR0, :CR1, :CR2, :CR3, :Cluster)
    @eval function resolve(k::$E{<:Tuple{Vararg{Symbol}}}, data)
        return $E(map(name -> Tables.getcolumn(data, name), clusters(k)))
    end
end
```

`resolve(CR1(:firm, :year), df)` is then `CR1((df.firm, df.year))`, which every
entry point accepts. The rows of the columns must line up with the rows of the model's
moment matrix, so a package that drops rows (for missing values, say) must drop the
same rows from the cluster columns.

A package can also accept column-named estimators directly in `vcov` by dispatching
on them:

```julia
function StatsAPI.vcov(k::CR1{<:Tuple{Vararg{Symbol}}}, m::MyModel)
    return vcov(resolve(k, m.data), m)
end
```

### `vcov(estimator)` and `VcovSpec`

With a single argument, `vcov(estimator)` computes nothing: it returns a
[`VcovSpec`](@ref) whose `source` field is the estimator. CovarianceMatrices defines no
operation on a `VcovSpec`. A package that supports `model + vcov(estimator)` defines
the `+` method, reads `v.source`, resolves column names against its own data, and
attaches the result to the model:

```julia
function Base.:+(m::MyModel, v::VcovSpec)
    k = resolve(v.source, m.data)
    return with_vcov_estimator(m, k)   # package-specific
end
```

## Building Moment Matrices in a Modeling Package

A package can form the moment matrix of its own models (to avoid materializing the
design matrix, say) and pass it to `aVar` on a matrix. `aVar` on a matrix accepts
only the estimators that need no fitted model: `HR0`/`HC0`, `CR0`, `Cluster`, the
long-run estimators, and `DriscollKraay` with `correction = :HC0`. The package applies
the model-based adjustments itself, with these public functions:

| Function | Use |
|----------|-----|
| [`CovarianceMatrices.residual_adjustment`](@ref)`(k, model)` | the factor of `HR1`–`HR5` that multiplies the residuals |
| [`CovarianceMatrices.numobs`](@ref)`(model)` | the `n` of the finite-sample corrections; define it if `modelmatrix` is expensive |
| [`CovarianceMatrices.bandwidthweights`](@ref)`(k, X)` | the bandwidth-selection weights of the design matrix, passed as `aVar(...; weights)` |
| [`CovarianceMatrices.clusters`](@ref)`(k)`, [`CovarianceMatrices.groups`](@ref), [`CovarianceMatrices.nclusters`](@ref) | the clusters of a cluster estimator, for building `CR0` on them |
| [`CovarianceMatrices.timekernel`](@ref), [`CovarianceMatrices.timeindex`](@ref), [`CovarianceMatrices.unitindex`](@ref) | the parts of a `DriscollKraay` estimator, for rebuilding it without its correction |
| [`estimator`](@ref CovarianceMatrices.estimator(::CachedCR))`(k)`, [`CovarianceMatrices.cache`](@ref)`(k)` | the parts of a `CachedCR` |

For `HC3` on the least-squares model `ols` defined above, with the bread `inv(X'X)`:

```@example iface
using CovarianceMatrices: residual_adjustment, momentresiduals, bandwidthweights

Xo = modelmatrix(ols)
a = residual_adjustment(HC3(), ols)
M = Xo .* (a .* momentresiduals(ols))
B = inv(Xo' * Xo)
n = size(Xo, 1)
n .* B * aVar(HC0(), M; demean = false) * B ≈ vcov(HC3(), ols)
```

A data-driven HAC bandwidth is selected from the columns of the moment matrix, giving
weight 0 to a constant column. The intercept column of a regression's moment matrix
is the residuals, which are not constant, so the weights come from the design matrix:

```@example iface
k = Bartlett{Andrews}()
M = Xo .* momentresiduals(ols)
A = aVar(k, M; demean = false, weights = bandwidthweights(k, Xo))
n .* B * A * B * n / (n - 2) ≈ vcov(k, ols)
```

The factor `n / (n - K)` is the degrees-of-freedom correction that `vcov` applies to
the long-run estimators on a `RegressionModel`.

## Testing Your Implementation

```julia
using Test

@testset "MyModel Interface" begin
    # Create a model instance
    model = MyModel(...)

    # Test required methods exist and return correct types
    @test hasmethod(CovarianceMatrices.momentmatrix, (typeof(model),))
    @test hasmethod(StatsAPI.coef, (typeof(model),))
    @test hasmethod(StatsAPI.nobs, (typeof(model),))

    # Test dimensions
    Z = CovarianceMatrices.momentmatrix(model)
    β = StatsAPI.coef(model)
    T = StatsAPI.nobs(model)

    @test size(Z, 1) == T  # Rows = observations
    @test size(Z, 2) >= length(β)  # Cols ≥ parameters

    # Test variance computation works
    @test_nowarn vcov(HC3(), model)
    @test_nowarn stderror(HC3(), model)

    # Test variance forms if applicable
    if model isa CovarianceMatrices.MLikeModel
        @test_nowarn vcov(Information(), model)
    end
end
```

## Summary

1. **Minimal effort**: Implement 3 methods, get all estimators
2. **Maximum flexibility**: Duck typing means no forced inheritance
3. **Optional features**: Add methods incrementally as needed
4. **Type safety**: Use abstract types for compile-time checks
5. **Full power**: Access to all HAC, HC, CR, VARHAC, etc. estimators

The interface is designed to be **easy to implement** but **powerful to use**. Start minimal, extend as needed.

## Further Reading

- [Introduction & Mathematical Foundation](../introduction.md): Theory behind the estimators
- [GLM Tutorial](glm_tutorial.md): See how GLM.jl integration works
- [API Reference](../api.md): Complete API documentation
- Source code: `src/model_interface.jl` and `src/api.jl`
