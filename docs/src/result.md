# The Returned CovarianceMatrix

`aVar(k, X)` and `vcov(k, model)` return a [`CovarianceMatrix`](@ref). It is an
`AbstractMatrix` that holds three things: the estimate, the estimator that produced
it, and the quantities the estimator chose while computing it, such as a
data-driven bandwidth or the lag orders selected by `VARHAC`.

```@example result
using CovarianceMatrices, GLM, DataFrames, RDatasets, LinearAlgebra

capm = dataset("Ecdat", "Capm")
capm.RFoodX = capm.RFood .- capm.RF
model = lm(@formula(RFoodX ~ RMRF), capm)

V = vcov(Bartlett{Andrews}(), model)
```

## Using It as a Matrix

Indexing, `size`, `diag`, `inv`, `\`, `cholesky` and the other functions defined
for an `AbstractMatrix` work on `V`. The standard errors are the square roots of
the diagonal:

```@example result
sqrt.(diag(V)) ≈ stderror(Bartlett{Andrews}(), model)
```

There is no `stderror(V)` method; use `stderror(k, model)` or `sqrt.(diag(V))`.

The results of these functions are plain matrices, as are the results of
arithmetic such as `2V` or `V + V`: the stored estimator and quantities describe
`V` and nothing derived from it. `parent(V)` returns the stored matrix without
copying it, and `Matrix(V)` returns a copy.

The stored matrix is exactly symmetric, so `cholesky(V)` works without wrapping
`V` in `Symmetric`.

## What It Stores

`estimator(V)` returns the estimator that produced `V`:

```@example result
estimator(V)
```

[`information`](@ref)`(V)` returns a `NamedTuple` of the quantities the estimator
chose. Its fields depend on the estimator:

```@example result
information(V)
```

| Estimator | Fields of `information(V)` | Read with |
|-----------|----------------------------|-----------|
| `Bartlett`, `Parzen`, `QuadraticSpectral`, `Truncated`, `TukeyHanning`; `DriscollKraay` | `bandwidth`, `kernelweights` | [`bandwidth`](@ref)`(V)`, [`kernelweights`](@ref)`(V)` |
| `VARHAC` | `AICs`, `BICs`, `order_aic`, `order_bic`, `order` | [`CovarianceMatrices.order`](@ref)`(V)` and the functions of the same names |
| `CR0`–`CR3`, `Cluster` | `nclusters` | `information(V).nclusters` |
| `HR0`–`HR5` (`HC0`–`HC5`), `HC4m`, `Uncorrelated`, `EWC`, `UniformSmoother`, `TriangularSmoother` | none | — |

For a HAC kernel, `bandwidth(V)` is the bandwidth the estimate used: the one given
to a fixed-bandwidth kernel such as `Bartlett(5)`, or the one selected by a
data-driven kernel such as `Bartlett{Andrews}()`. `kernelweights(V)` holds the
weight of each column in the bandwidth selection; on a fitted model with an
intercept, as above, the intercept's weight is zero. A fixed bandwidth involves no
selection, so `kernelweights(V)` is `nothing`:

```@example result
V_fixed = vcov(Bartlett(5), model)
(bandwidth = bandwidth(V_fixed), kernelweights = kernelweights(V_fixed))
```

For `VARHAC`, `order(V)` holds the selected lag order of each column of the
moment matrix (one per coefficient on a fitted model), by the criterion the
estimator was built with; `order_aic(V)` and `order_bic(V)` hold the
orders each criterion selects, and `AICs(V)` and `BICs(V)` the criterion values
over the lag orders searched. These functions are not exported:

```@example result
V_varhac = vcov(VARHAC(), model)
CovarianceMatrices.order(V_varhac)
```

For a cluster estimator, `nclusters` holds the number of clusters in each
clustering dimension, one entry per dimension. `bandwidth(V)` and
`kernelweights(V)` return `nothing` for an estimator that does not store them.

`EWC`, `UniformSmoother` and `TriangularSmoother` select nothing during
estimation: their tuning parameter is part of the estimator, so `estimator(V)`
shows it.

For a model built on `MLikeModel` or `GMMLikeModel`, `vcov(ve, form, model)`
stores the estimator `ve` and the quantities of its estimate of the moment
covariance, and `vcov(Information(), model)` stores `Information()` and nothing
else.
