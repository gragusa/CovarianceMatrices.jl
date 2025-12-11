# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.30.2] (Unreleased)

### Added

- Add more tests for EWC

## [0.30.2]

### Fixed

- **EWC Variance Estimator Bugs**: Fixed two critical bugs in the EWC (Equal Weighted Cosine) estimator:
  - Added missing `residual_adjustment` method for EWC, which caused errors when using `vcov(EWC(B), model)` with regression models
  - Fixed double scaling bug where variance was divided by n twice (once in `Λ!` function and again in `aVar`), resulting in variance estimates that were too small by a factor of n

### Added

- Added Monte Carlo coverage tests to validate EWC estimator produces correct confidence interval coverage (~91% with AR(1) errors)

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
