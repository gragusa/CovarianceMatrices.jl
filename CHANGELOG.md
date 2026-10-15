# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Inputs that 0.32.1 accepted with a deprecation warning now throw an
  `ArgumentError`:
  - `aVar` with `HR1`–`HR5`, `HR4m` (and their `HC` aliases) or `CR1`–`CR3`,
    alone or inside `CachedCR`, on a matrix. Their finite-sample corrections
    need a fitted model; use `HR0`/`HC0` or `CR0`/`Cluster` on a matrix. This
    includes the `vcov(ve, form, model)` methods for `MLikeModel` and
    `GMMLikeModel`, which pass a moment matrix.
  - `prewhite = true` with an estimator that does not prewhiten (anything other
    than HAC kernels, `EWC`, the smoothers, and `DriscollKraay` with a HAC
    kernel).
  - Keywords that `aVar(::VARHAC, X)` does not use.
  - `VARHAC(AICSelector(), FixedLags(k))` and `VARHAC(BICSelector(), FixedLags(k))`;
    use `VARHAC(FixedLags(k))`.
  - Non-integer lag orders in `FixedLags`, `SameLags` and `DifferentOwnLags`.
    Integer-valued numbers such as `3.0` are still accepted.

### Removed

- The forms deprecated in 0.32.0:
  - `k.kw` and `k.wlock` on a HAC kernel; use `kernelweights(aVar(k, X))`.
  - `AICs`, `BICs`, `order`, `order_aic` and `order_bic` on a `VARHAC`
    estimator; call them on the result of `aVar(k, X)`.
  - A numeric `scale` keyword; `scale` accepts only a `Bool`, and a divisor is
    passed as `scaleby`.
  - `optimal_bandwidth(k::MomentSmoother, T)`; use `optimalbw(k, G)` with the
    moment matrix `G`.

## [0.32.1]

### Fixed

- The `CR0` docstring showed a `G/(G-1)` factor that `CR0` does not apply.

- `aVar`, `vcov`, `stderror`, `nclusters`, `CachedCR` and `CachedCRModel` throw
  an `ArgumentError` explaining that column names must be resolved by a
  modeling package when given a column-named cluster estimator such as
  `CR1(:firm)`, instead of a `MethodError` from an internal function.

- Estimators print to the given `IO` stream, so `repr`, `string` and
  interpolation capture the whole output, and the header of a displayed
  `CovarianceMatrix` is no longer split across streams. Estimators print as
  constructor syntax that rebuilds an equal estimator, for example
  `Bartlett(5.0)`, `Bartlett{Andrews}()`, `CR0([1, 1, 2, 2])`,
  `VARHAC(AICSelector(), SameLags(4))` and
  `DriscollKraay(Bartlett(3.0), tis = [...], iis = [...])`. Cluster assignments
  print as integer group codes. `CachedCRModel` prints a summary of its cache.

- Every `aVar` and `vcov` method returns a `CovarianceMatrix`, as 0.32.0
  announced. `aVar`/`vcov` with a `CachedCRModel`, `vcov` with a
  `DriscollKraay` estimator on a model, and the four `vcov` methods for
  `MLikeModel` and `GMMLikeModel` returned a plain `Matrix`.
  - `aVar` with a `DriscollKraay` estimator reports the inner kernel's
    bandwidth and kernel weights, so `bandwidth(V)` works for a data-driven
    kernel inside it.
  - Cluster-robust estimates record the number of clusters in each dimension
    in `information(V).nclusters`.
  - `vcov(ve, Information(), ::MLikeModel)` does not use `ve`; its result
    records `Information()` as the estimator.

- Invalid inputs throw errors that name the problem:
  - `Truncated{NeweyWest}(bw)` and `TukeyHanning{NeweyWest}(bw)` throw, as the
    forms without a bandwidth already did; the message names the kernels the
    Newey-West rule supports.
  - `Bartlett{Fixed}()` and the other kernels with `Fixed` and no bandwidth
    throw an `ArgumentError` instead of a `MethodError`.
  - A cluster variable whose length differs from the number of observations
    throws a `DimensionMismatch` on every cluster path. On the model path it
    read out of bounds.
  - Cluster checks on the matrix path throw `DimensionMismatch` or
    `ArgumentError` instead of `AssertionError`.
  - A fixed bandwidth that is `NaN`, infinite or negative throws an
    `ArgumentError` at construction, and so does a zero bandwidth for
    `QuadraticSpectral`. A zero bandwidth for the other kernels uses no lags.
  - `CR1` with a single cluster, and `HC1`/`HR1` or the HAC degrees-of-freedom
    correction with no more observations than coefficients, throw an
    `ArgumentError` instead of returning an infinite variance.
  - `maxlags` on a `VARHAC` with `AutoLags` and no data dimensions throws an
    `ArgumentError`.
  - The model-interface checks report a missing `coef`, `nobs` or
    `momentmatrix` method, and let errors raised inside an implemented
    method propagate unchanged.

- `prewhite = true` reaches `EWC` and the moment smoothers, which implement
  prewhitening; `aVar` ignored it for them.

- `VARHAC` warns when a lag regression or `I - ΣAⱼ` is singular and a
  pseudo-inverse is used in its place.

- `DriscollKraay`:
  - `aVar(k, X)` is on the same per-observation scale as every other
    estimator: it divided the kernel sum over period sums by `n·T` (by `T`
    with `scale = false`) instead of by `n`. Values from `aVar` change by a
    factor `T`; `vcov(k, model)` results are unchanged.
  - Periods are ordered by sorting `tis`. They were ordered by first
    appearance in the data, so rows not sorted by time gave a wrong estimate.
  - `demean = false` is honored; the period sums were always demeaned.
  - With a data-driven kernel on a model, the bandwidth selection gives the
    intercept weight 0, as for the kernel used directly. This changes the
    selected bandwidth.
  - `prewhite = true` prewhitens the series of period sums.
  - Any `AbstractMatrix` is accepted. A `tis` whose length differs from the
    number of rows, `tis` and `iis` of different lengths, and a repeated
    (period, unit) pair throw.

- Automatic differentiation with `ForwardDiff` through `aVar`:
  - `EWC` returned an estimate scaled by `1/√T` for `Dual` input, so
    gradients were wrong, and failed with `prewhite = true`. `EWC` now uses
    the same code for every element type; the separate `ForwardDiff`
    extension is gone.
  - `VARHAC` with a nonzero lag order failed for `Dual` input.
  - Prewhitening failed for `Dual` input.
- `CR0`–`CR3` and `Cluster` accept any `AbstractVector` or `AbstractMatrix`,
  including sparse matrices and views. Sparse input took no path through
  `aVar`, although the `SparseArrays` extension implements it.
- `VARHAC(FixedSelector(), strategy)` and `VARHAC(:fixed)` select the lag
  order by BIC, but `order` reported the AIC selection; it now reports the BIC
  orders the estimate is computed from. `VARHAC(FixedLags(3))` prints as
  `VARHAC(FixedLags(3))`.
- `FixedLags` and `SameLags` throw for an order below 1, and
  `DifferentOwnLags` throws for a negative order or a vector that is not
  `[own, cross]`, at construction. These inputs previously failed inside
  `aVar` or were accepted silently. The `DifferentOwnLags` docstring described
  the vector as one lag per variable.

### Added

- `DriscollKraay(K; tis, iis, correction)` carries the finite-sample
  correction (`:HC0`, `:HC1` or `:sss`), which `vcov(k, model)` applies.
  `aVar` on a moment matrix throws for a correction other than `:HC0`.

- `Cluster(:firm)` and `Cluster(a, b, ...)`, matching the `CR0`–`CR3`
  constructors.

- The interface tutorial documents column-named cluster estimators and
  `VcovSpec` for modeling packages.

- `Fixed(bw)` is a bandwidth value: `Bartlett(Fixed(5.0))` is the same kernel
  as `Bartlett(5.0)`, and `bandwidth(Fixed(5.0)) == 5.0`. `Fixed()` without a
  bandwidth is no longer defined.

- `optimalbw(k::MomentSmoother, G)` takes the moment matrix, as for HAC
  kernels.

- `CachedCR(k, model)` caches the cluster indices and the `CR2`/`CR3`
  leverage adjustments of a regression model. Every `CachedCR` works on both
  moment matrices and models: `CachedCR(k, ncols)` on a model gives the
  results of `k`, and a model-built cache checks that the model matches it.
  Two `CachedCR` values compare equal (and hash equally) when they wrap equal
  estimators, whatever their caches.

- A documented, `public` protocol for `RegressionModel` types other than GLM
  fits: `bread`, `momentresiduals`, `leverage` and `mask`, together with
  `coef`, `modelmatrix` and (for `CR2`/`CR3`) `weights`. `momentresiduals`
  falls back to `StatsAPI.residuals`, `leverage` to `StatsAPI.leverage`, and
  `mask` to all coefficients estimable, so an unweighted least-squares model
  needs only `bread` beyond the StatsAPI functions. Models that define the
  internal `_residuals` keep working. The number of observations is the number
  of rows of `modelmatrix`.

- `vcov(Information(), model)` and `stderror(Information(), model)` for an
  `MLikeModel`.

- `Cluster` is a subtype of `CR` and works on fitted models, where it equals
  `CR0`, as `Uncorrelated` equals `HC0`. Its results record
  `information(V).nclusters`, and it can be wrapped in `CachedCR`.

- `CR` is exported. `LagStrategy` and `Clustering` (the type stored in the `g`
  field of cluster estimators) are `public`.

### Deprecated

- `vcov(k::DriscollKraay, model; type = ...)`; use
  `DriscollKraay(...; correction = ...)`.

- `prewhite = true` with an estimator that does not prewhiten, and keywords
  that `aVar(::VARHAC, X)` does not use. Both are ignored with a deprecation
  warning and will throw in a future release.

- `String(T)` for estimator types such as `String(Bartlett)`.

- VARHAC: `FixedSelector` and `VARHAC(:fixed)` (use `BICSelector()`/`:bic`, or
  `VARHAC(FixedLags(k))` for a fixed order); a selector passed together with
  `FixedLags`; non-integer lag orders, which are rounded; and the `T` keyword,
  which has no effect. A selector with `FixedLags` and non-integer lags will
  throw in a future release.

- A bandwidth given to a data-driven kernel, as in `Bartlett{Andrews}(5.0)`.
  The value is ignored (the kernel equals `Bartlett{Andrews}()`); this form
  will throw in a future release.

- `kernelweights(k, X)` on an estimator; `kernelweights` reads the weights
  from a result, `kernelweights(aVar(k, X))`.

- `optimalbw(k::MomentSmoother, T::Integer)`; use `optimalbw(k, G)` with the
  moment matrix `G`.

- `vcov(ve, Information(), model)` and `stderror(ve, Information(), model)` for
  an `MLikeModel`, which do not use `ve`; omit the estimator.

- `CachedCRModel` is a deprecated alias of `CachedCR`; `CachedCRModel(k, model)`
  returns `CachedCR(k, model)`, which prints as `CachedCR(...)`.

- `aVar(k, X)` on a matrix with `HC1`–`HC5`, `HC4m`, `HR1`–`HR5`, `HR4m` or
  `CR1`–`CR3`. Their corrections are formed from a fitted model, so on a matrix
  they equal `HC0` or `CR0`. Use `HC0`, `CR0` or `Cluster` with a matrix, or pass
  a fitted model; this form will throw in a future release.

- `a𝕍ar`; use `aVar`.

## [0.32.0]

### Breaking

- `aVar` and `vcov` return a `CovarianceMatrix` instead of a plain `Matrix`. It
  subtypes `AbstractMatrix`, so indexing, arithmetic, `inv`, `\`, `diag` and
  factorizations work unchanged; operations that do not preserve the covariance
  interpretation return a plain matrix. `parent(V)` recovers the underlying
  matrix.

- Estimators are immutable specifications and no longer carry fit state. HAC
  kernels previously held `bw`/`kw`/`wlock` vectors and `VARHAC` held its AIC/BIC
  tables and selected orders, so reusing an estimator silently rewrote the
  bandwidth behind an earlier estimate, `==` reported two kernels equal while
  their bandwidths differed, and concurrent use raced. What was selected from the
  data now lives on the returned `CovarianceMatrix` and is reached through
  `estimator`, `bandwidth`, `kernelweights`, `information`, and for `VARHAC`
  `order`, `order_aic`, `order_bic`, `AICs` and `BICs`.

- The `scale` keyword of `aVar` and `vcov` is a `Bool` switch only. A divisor is
  passed as `scaleby`, which supersedes the switch so the variance is divided
  once:

      aVar(k, X; scale = false)    # no division
      aVar(k, X; scaleby = n - p)  # divide by an explicit value

  Divisors may be any positive finite real; a degrees-of-freedom correction is
  not generally an integer. Passing a number as `scale` still works and warns.

- `aVar(::VARHAC, ::AbstractMatrix)` honors `scale`. Both arms of the
  `scale === false` branch returned the same matrix, so a caller asking for
  unscaled output received scaled output. VARHAC estimates the spectral density
  at frequency zero, which already carries the per-observation scaling the other
  estimators reach through `scale = true`; reaching the unscaled convention
  multiplies by the number of observations rather than dividing.

- `aVar` on a matrix whose element type is not `Real` raises a `MethodError`. The
  `AbstractMatrix` fallback was shadowed for every real element type and recursed
  infinitely for any other.

### Deprecated

- `k.kw` and `k.wlock` on a HAC kernel. Kernel weights belong to an estimate:
  use `kernelweights(aVar(k, X))`.
- `AICs(k)`, `BICs(k)`, `order(k)`, `order_aic(k)` and `order_bic(k)` on a
  `VARHAC` estimator. Call them on the result of `aVar(k, X)` instead.
- A numeric `scale` keyword. Use `scaleby`.
- `optimal_bandwidth(k::MomentSmoother, T)`. Bandwidth selection is spelled
  `optimalbw` for every estimator family.

### Added

- `CovarianceMatrix` and its accessors `estimator`, `bandwidth`,
  `kernelweights` and `information` are exported.
- A `PDMats` extension: `PDMat(V::CovarianceMatrix)` gives access to the PDMats
  interface, and `isposdef(V)` checks first. Not every estimator is positive
  definite — the `Truncated` kernel is consistent without being positive
  semi-definite, the multiway cluster-robust estimator can be indefinite in
  finite samples, and a rank-deficient model produces `NaN` entries.
- `optimalbw` on a fixed-bandwidth kernel returns the configured bandwidth
  instead of throwing a `MethodError`, so a call site can switch between a fixed
  and a data-driven kernel unchanged.
- `Cluster`, the clustering analogue of `Uncorrelated` for the matrix / moment
  interface. `aVar(Cluster(groups), X)` returns the raw cluster long-run variance
  with no finite-sample correction, since degrees-of-freedom and leverage factors
  require a fitted model's design matrix. For cluster-robust regression standard
  errors use `CR0`–`CR3` with a model; `Cluster` on a model raises an
  `ArgumentError` pointing to them.

### Changed

- Eighteen documented non-exported names are declared `public` on Julia 1.11 and
  later: the result and estimator accessors `AICs`, `BICs`, `maxlags`,
  `nclusters`, `order`, `order_aic` and `order_bic`; the kernel struct names
  `BartlettKernel`, `ParzenKernel`, `QuadraticSpectralKernel`, `TruncatedKernel`
  and `TukeyHanningKernel` behind the exported aliases; the cluster caches
  `CRCache` and `CRModelCache`; and the abstract supertypes `BandwidthType`,
  `CR` and `LagSelector`. Every other non-exported name is an implementation
  detail and may change in any release.
- The positional `DriscollKraay` constructor accepts identifiers of any type.
  It was pinned to `AbstractArray{<:AbstractFloat}`, and integer identifiers
  fell through to the default inner constructor, producing an estimator that
  failed later inside `aVar` with "type Array has no field ngroups". Both call
  forms now share one coercion path, and `tis` and `iis` are required arguments
  that each report their own absence.

### Bug Fixes

- `aVar(Uncorrelated(), X)` and `vcov(Uncorrelated(), model)` work instead of
  throwing a `MethodError`. `Uncorrelated` now matches HC0/HR0 (White's
  estimator) on both a matrix and a fitted model.
- `vcov` and `stderror` work on a fitted model for `VARHAC` and for the
  `UniformSmoother`/`TriangularSmoother` moment smoothers, which previously
  threw a `MethodError`.
- `workingoptimalbw` for fixed-bandwidth kernels threw a `TypeError` on every
  call: it wrote `Matrix{eltype{m}}` with braces instead of parens.
- Removed `demeaner(k::CR, X)`, which called two functions the package does not
  define; every caller passes a matrix, not a `CR`.
- Removed two unreachable `scalevar!` methods: one was shadowed by an identical
  signature above it, and the other called `rdiv!` on an undefined variable.

## [0.31.0]

### Breaking

- `CRCache` and `CRModelCache` are no longer exported. Construct caches via the
  exported `CachedCR(k, ncols)` / `CachedCRModel(k, model)` instead.

### Added

- Exported the `Fixed` bandwidth marker so `HAC{Fixed}` kernels can be dispatched
  on directly (e.g. `Bartlett(4) isa HAC{Fixed}`).
- Value-based `Base.==` and matching `Base.hash` for estimator types, so equal
  specifications compare equal and work as `Dict` keys / `unique` elements.
  Equality reflects the estimator specification, not transient fit state.
- `vcov(::DriscollKraay, model)` accepts a `type` argument (`:HC0`, `:HC1`,
  `:sss`) for the finite-sample corrections of R's `plm::vcovSCC`.

### Bug Fixes

- `vcov(::DriscollKraay, model)` now works for regression models and reproduces
  `plm::vcovSCC`. It previously threw a `MethodError`, and the underlying
  sandwich scaled by the observation count `n` rather than the number of time
  periods `T`.

- GMM variance formulas for suboptimal weight matrix:
  - `_compute_gmm_information_weighted` was computing `inv(G'WΩ⁻¹WG)` instead of the correct sandwich `(G'WG)⁻¹ G'WΩWG (G'WG)⁻¹`. 
  - `_compute_gmm_misspecified` with explicit W used `Ω⁻¹` in the meat instead of `Ω`. The efficient GMM paths (W=nothing) were already correct.
- Sign handling in multi-way cluster variance 
  - Removed incorrect `(-1)^(len-1)` sign factors from `_avar_tuple_impl` in `CR.jl`
- Corrected the `UniformSmoother`/`TriangularSmoother` bandwidth validation
  messages: negative `m_T` now reports "must be non-negative" (the contract is
  `m_T ≥ 0`), and the non-integer check reports "must be an integer"

### Changed

- GMM API 
  - Weight matrix is now resolved via `weight_matrix(model)` before dispatching to compute functions; simplified Misspecified GMM path
- CR2/CR3 leverage adjustments 
  - Deduplicated code via `_leverage_transform` dispatch, unifying `residual_adjustment` and `_compute_leverage_adjustments` for CR2/CR3
- Multi-vector Clustering constructor 
  - Delegates to single-vector constructors and merge, avoiding `NTuple{N, Any}` dict keys
- Removed unused `ncombinations` variable from `CRCache`
- Improves smoothing functions
- Removed the duplicate `HR` export and the unused `CrossSectionEstimator`
  abstract type
- `vcov` now throws a clear `ArgumentError` naming `jacobian_momentfunction` when
  that required GMM hook is missing, instead of a low-level `MethodError`
- Changed `order_aic` and `order_bic` fields from `Vector{Int}` to `Array{Int}` to support both `SameLags` (Vector) and `DifferentOwnLags` (Matrix) strategies
- VARHAC optimization - Moved `delag(X, kk)` call inside the kk > 0 condition in `_var_selection_ownlag`

### Documentation

- Simplified documentation index page
- Expanded asymptotic variance theory in introduction 
- Updated GMM docstrings to reflect corrected formulas
- Documented the layered API: the `aVar(k, matrix)` matrix core, the
  `vcov(k, model)` model protocol, and the `model + vcov(estimator)` wrapper
- Documented the recommended VARHAC construction form and its convenience aliases

###  CI/Infrastructure

- Added CompatHelper, TagBot, and benchmark GitHub workflows
- Updated CI workflow configuration
- Added .pre-commit-config.yaml
- Added benchmark infrastructure

### Tests

- Added analytical tests for all four GMM variance formulas
- Added two-way cluster variance regression test 
- Added more tests for smoothed moments
- Added VARHAC tests

## [0.30.4]

### Added

- VcovSpec Wrapper (#78): New VcovSpec{T} type enabling model + vcov(estimator) syntax for robust standard errors (e.g., `model + vcov(HC3())` or `model + vcov(CR1(:firm)))`.
- Cached Cluster-Robust Estimators (#72): New CachedCR and CRCache types for fast repeated cluster-robust variance calculations with preallocated buffers and precomputed cluster indices.
- `CachedCR(k, ncols)` — For general moment matrices
- `CachedCRModel(k, model)` — For RegressionModel/GLM with precomputed leverage adjustments
- Internal Clustering Implementation (#72): Lightweight `Clustering` struct replacing `GroupedArrays.jl` dependency.

  Changed

  - CR estimators (`CR0`, `CR1`, `CR2`, `CR3`) now use internal Clustering type instead of GroupedArray.

  Fixed

  - Implicit Imports (#77): Fixed implicit import issues for cleaner namespace management.

  Tests

  - Improved test coverage (#76).


## [0.30.3]

### Added

- Add more tests for `EWC`

- Added symbol constructor for `CR` type variance. E.g., `CR0(:state)` which gives a type `CR0{Tuple{Symbol}}`. Useful for specifying cluster variables using symbols, allowing models that support it to materialize the actual type later. 


## [0.30.2]

### Fixed

- **EWC Variance Estimator Bugs**: Fixed two critical bugs in the EWC (Equal Weighted Cosine) estimator:
  - Added missing `residual_adjustment` method for EWC, which caused errors when using `vcov(EWC(B), model)` with regression models
  - Fixed double scaling bug where variance was divided by n twice (once in `Λ!` function and again in `aVar`), resulting in variance estimates that were too small by a factor of n

### Added

- Added Monte Carlo coverage tests to validate EWC estimator produces correct confidence interval coverage

## [0.30.1]

### Added

- Fix bug where Julia was restricted to v1.11 instead of v1.10 (lts).

- **Aqua.jl Quality Assurance**: Added comprehensive code quality tests using Aqua.jl, addressing issue requirements for automated detection of undefined exports, stale dependencies, and other common package issues

## [0.30.0]

### Added

- **Smith's Smoothed Moments Implementation**: Full, optimized implementation with kernel-based approach
  
- **Alternative Constructor Syntax for HAC Kernels**: Support both `Kernel(BandwidthType)` and `Kernel{BandwidthType}()` syntax
  - `Bartlett(Andrews)` and `Bartlett{Andrews}()` are now functionally equivalent
- **Comprehensive Documentation**:
  - Complete docstrings for all major estimator types with LaTeX-formatted mathematical foundations
  
  
### Changed

- **Type Hierarchy Refactoring**: Fundamental restructuring to improve semantic clarity
  - `AVarEstimator` → `AbstractAsymptoticVarianceEstimator`
  - Introduced `Uncorrelated` type for i.i.d. errors (semantic alternative to `HR0` for MLE/GMM)
  - Introduced `Correlated` abstract parent for all correlation-based estimators
  - `EWC`, `VARHAC`, `DriscollKraay`, and `SmoothedMoments` now correctly inherit from `Correlated`
- **Improved Interface**: General interface to `RegressionModel` for better ecosystem compatibility
- **CI/CD Improvements**:
  - Refactored CI workflow for stability and concurrency
  - Updated job matrix for Julia versions and OS compatibility
  - Applied `JuliaFormatter` (sciml style) across the codebase

### Removed

- Deleted unused files and legacy code

## [0.22.0] - Previous Release

Initial baseline for this changelog.

[0.22.0]: https://github.com/gragusa/CovarianceMatrices.jl/releases/tag/v0.22.0
