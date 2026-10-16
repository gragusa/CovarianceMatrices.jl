# Performance

`CovarianceMatrices.jl` is built for speed: robust variance estimation is often
the inner loop of bootstrap inference, simulation studies, and iterative
estimators, where the covariance is computed thousands of times. This page
compares it against R's [`sandwich`](https://cran.r-project.org/package=sandwich)
package on identical data and estimators, and describes how to reproduce and
extend the measurements.

## Comparison with R's `sandwich`

The tables below report **median** wall-clock time over 50 evaluations. Julia
times use `BenchmarkTools.@benchmark`; R times use `microbenchmark`. Both run the
same estimator on the same data.

!!! note "Measurement environment"
    Apple M3 Max, macOS 26.5, single-threaded. Julia 1.12 with
    CovarianceMatrices.jl 0.31; R 4.6 with sandwich 3.1.1. The Quadratic
    Spectral rows use Julia 1.13 and CovarianceMatrices.jl 0.33. Absolute numbers are machine-dependent; the *ratios* are the
    portable takeaway. Run the scripts below to get figures for your own
    hardware.

### HAC (long-run variance, Andrews bandwidth)

A `T × k` matrix of moment contributions, Andrews automatic bandwidth, no
prewhitening (`aVar(Bartlett{Andrews}(), Z; prewhite = false)` against
`sandwich::lrvar(Z, type = "Andrews", kernel = "Bartlett", adjust = FALSE)`).

| Kernel | T | k | CovarianceMatrices.jl | sandwich | Speedup |
|--------|---|---|----------------------:|---------:|--------:|
| Bartlett           | 1000  | 10 | 0.032 ms  | 8.62 ms   | ~270× |
| Parzen             | 1000  | 10 | 0.049 ms  | 8.65 ms   | ~175× |
| Quadratic Spectral | 1000  | 10 | 0.147 ms  | 67.5 ms   | ~460× |
| Bartlett           | 10000 | 10 | 0.314 ms  | 66.2 ms   | ~210× |
| Parzen             | 10000 | 10 | 0.495 ms  | 66.6 ms   | ~135× |
| Quadratic Spectral | 10000 | 10 | 1.42 ms   | 761.9 ms  | ~540× |

The Quadratic Spectral kernel has unbounded support, so every lag enters the
estimate. CovarianceMatrices.jl computes it as `Z' K Z` with the `T × T`
Toeplitz matrix of kernel weights `K`, and computes `K Z` with FFTs in
`O(T log T)` time.

### Heteroskedasticity-robust (HC) standard errors

A linear model with `n = 5000`, `k = 6` coefficients
(`vcov(HC0(), fit)` against `sandwich::vcovHC(fit, type = "HC0")`).

| Type | CovarianceMatrices.jl | sandwich | Speedup |
|------|----------------------:|---------:|--------:|
| HC0 | 0.035 ms | 4.07 ms | ~115× |
| HC1 | 0.035 ms | 4.08 ms | ~115× |
| HC2 | 0.121 ms | 4.05 ms | ~33×  |
| HC3 | 0.099 ms | 3.94 ms | ~40×  |

HC2 and HC3 cost more than HC0/HC1 because they need the hat-matrix diagonal
(leverages).

<!-- ## Reproducing the comparison

Julia side:

```julia
using CovarianceMatrices, BenchmarkTools, Random
Random.seed!(123)

Z = randn(10000, 10)
@benchmark aVar(Bartlett{Andrews}(), $Z; prewhite = false)
```

R side:

```r
library(sandwich); library(microbenchmark)
set.seed(123)
Z <- matrix(rnorm(10000 * 10), 10000, 10)
microbenchmark(lrvar(Z, type = "Andrews", kernel = "Bartlett", adjust = FALSE),
               times = 50)
```

`sandwich::lrvar` returns the long-run variance of the column means, which is
`aVar(k, Z) / T`; the scalar does not affect timing.

## The benchmark suite

The repository ships an [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
suite at `benchmark/benchmarks.jl` covering HAC kernels (fixed, Newey-West and
Andrews bandwidths), the other long-run variance estimators (EWC, VARHAC, the
smoothers, Driscoll-Kraay), HC and cluster-robust estimators on fitted models
(one-way and two-way clustering, weights, a logit, `CachedCR`), and moment
smoothing. Its dependencies are listed in `benchmark/Project.toml`. The runner
script `benchmark/run_asv.jl` benchmarks a baseline revision against a candidate
and writes a markdown summary table. It needs AirspeedVelocity.jl in the default
environment:

```sh
# from a clone of the repository; compares HEAD against the working tree
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'
julia benchmark/run_asv.jl
```

Set `ASV_BASELINE` / `ASV_CANDIDATE` to compare specific revisions; the table is
written under `benchmark/results/`.

The `Benchmark` GitHub workflow runs the pull request's `benchmark/benchmarks.jl`
against both the base branch and the pull request, and posts a table comparing
them, so performance regressions are caught in review. Because the same script
runs against the base branch, it may use only API that exists there. The suite is the source of truth for the
relative performance of the package's own estimators; extend
`benchmark/benchmarks.jl` when adding new ones.

## Choosing a fast estimator

Statistical appropriateness comes first; among valid choices, these are the
fast paths:

```julia
# Cross-sectional, no leverage correction needed → HC0/HC1
ve = HC1()

# Time series, automatic and kernel-free → VARHAC (no bandwidth search)
ve = VARHAC()

# Time series, traditional kernel, fastest → fixed bandwidth
ve = Bartlett(b)

# Time series, data-driven bandwidth → Newey-West rule before Andrews
ve = Bartlett{NeweyWest}()   # cheaper than Bartlett{Andrews}()

# Panel with cross-sectional dependence → Driscoll-Kraay
ve = DriscollKraay(Bartlett(b), tis = time_ids, iis = unit_ids)

# Many clusters → CR1
ve = CR1(cluster_ids)
```

Two patterns dominate the cost of HAC estimation:

- **Bandwidth selection.** A fixed bandwidth (`Bartlett(b)`) skips selection
  entirely. The Newey-West rule is a closed-form formula; the Andrews rule fits
  a VAR(1) per series and is the most expensive. `VARHAC()` avoids kernel
  bandwidths altogether.
- **Kernel support.** Compactly supported kernels (Bartlett, Parzen, Truncated,
  Tukey-Hanning) only sum lags up to the bandwidth. The Quadratic Spectral
  kernel uses all lags; its cost grows as `T log T` because the lag sum is
  computed with FFTs.

## Numerical stability

The truncated and Tukey-Hanning kernels can give estimates that are not positive
semidefinite in finite samples, because their spectral windows take negative values.
Check the result when it matters:

```julia
function stability_check(Ω; tolerance = 1e12)
    κ = cond(Ω)
    κ > tolerance && @warn "Ill-conditioned covariance matrix (κ = $κ)"
    min_eig = minimum(eigvals(Symmetric(Ω)))
    min_eig < -1e-10 && @warn "Negative eigenvalue: $min_eig"
    return κ ≤ tolerance && min_eig ≥ -1e-10
end

Ω = aVar(TukeyHanning{Andrews}(), X)
stability_check(Ω)
```

The Bartlett, Parzen and quadratic spectral kernels give positive semidefinite
estimates, as do `VARHAC()`, the smoothed moment estimators (`UniformSmoother`,
`TriangularSmoother`) and `EWC`. A positive semidefinite estimate can still be close to
singular, so the condition-number check above remains useful.

## Profiling

To see where time goes in a single estimate:

```julia
using Profile
X = randn(10000, 10)
@profile for _ in 1:100; aVar(Bartlett{Andrews}(), X); end
Profile.print()
```

For HAC with automatic bandwidth, the selected bandwidth is stored on the kernel
after an estimate and can be read back with the (unexported) `bandwidth`
accessor, which returns it as a one-element vector:

```julia
k = Bartlett{Andrews}()
Ω = aVar(k, X)
CovarianceMatrices.bandwidth(k)   # e.g. [2.32], the Andrews-selected bandwidth
``` -->
