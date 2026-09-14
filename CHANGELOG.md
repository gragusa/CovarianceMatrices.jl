# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
