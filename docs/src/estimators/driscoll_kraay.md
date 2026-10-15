# Driscoll-Kraay Estimator

The Driscoll-Kraay estimator provides robust covariance estimation for panel data with both temporal and cross-sectional dependence. It extends HAC estimation to panels: observations may be correlated over time within a unit and, within a time period, across units.

## Mathematical Foundation

For a panel of $N$ units observed over $T$ periods, the estimator first sums the moment contributions across units within each period,

```math
h_t = \sum_{i=1}^{N} g_{it},
```

and then applies a HAC kernel to the time series $h_1, \dots, h_T$:

```math
\hat{\Omega}_{DK} = \hat{\Gamma}_0 + \sum_{j=1}^{T-1} k\!\left(\frac{j}{S_T}\right)\left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right],
\qquad \hat{\Gamma}_j = \frac{1}{n}\sum_{t=j+1}^{T} h_t h_{t-j}',
```

where $n$ is the number of observations. `aVar(dk, X)` returns $\hat{\Omega}_{DK}$: dividing by $n$ rather than $T$ puts it on the same per-observation scale as every other estimator, so the sandwich `vcov` is the same as for them. The moments are demeaned before the period sums are formed (unless `demean = false`), and periods are ordered by sorting the time identifiers.

Summing over units within a period leaves the cross-sectional correlation unrestricted, so the estimator is robust to arbitrary spatial dependence at each date. Only a single (temporal) kernel and bandwidth are needed.

## Core Type

```@docs
DriscollKraay
```

## Usage

The estimator combines a HAC kernel with the panel's time and unit identifiers. The Grunfeld data track investment for ten firms over twenty years; both serial correlation within firms and common shocks across firms are present.

```@example dk
using CovarianceMatrices, GLM, RDatasets, LinearAlgebra

grunfeld = RDatasets.dataset("plm", "Grunfeld")
model = lm(@formula(Inv ~ Value + Capital), grunfeld)

dk = DriscollKraay(Bartlett{Andrews}(), tis=grunfeld.Year, iis=grunfeld.Firm)
stderror(dk, model)
```

Compared with the classical standard errors, and with firm-level clustering:

```@example dk
using DataFrames
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    cluster_firm = stderror(CR1(grunfeld.Firm), model),
    driscoll_kraay = stderror(dk, model),
)
```

## Finite-Sample Corrections

The `correction` keyword of `DriscollKraay` selects a finite-sample correction that `vcov(dk, model)` applies for a `RegressionModel`, reproducing the scalar `type` options of R's `plm::vcovSCC`. With `n` observations, `k` coefficients, and `T` time periods:

| `correction` | Factor $c$ | Description |
|--------|------------|-------------|
| `:HC0` (default) | $1$ | No correction |
| `:HC1` | $n/(n-k)$ | Degrees-of-freedom correction |
| `:sss` | $\dfrac{n-1}{n-k}\cdot\dfrac{T}{T-1}$ | Stata-style small-sample correction |

The correction needs the number of coefficients, so `aVar` on a bare moment matrix throws an `ArgumentError` for an estimator with a correction other than `:HC0`.

```@example dk
dk_fixed(correction) = DriscollKraay(Bartlett(5), tis=grunfeld.Year, iis=grunfeld.Firm; correction)
DataFrame(
    coef = coefnames(model),
    HC0 = stderror(dk_fixed(:HC0), model),
    HC1 = stderror(dk_fixed(:HC1), model),
    sss = stderror(dk_fixed(:sss), model),
)
```

A bandwidth of `Bartlett(b)` corresponds to `plm::vcovSCC(..., maxlag = b - 1)`. The leverage-based `:HC2`/`:HC3`/`:HC4` options of `plm::vcovSCC` are not supported.

## References

- Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data". *Review of Economics and Statistics*, 80(4), 549-560.
- Hoechle, D. (2007). "Robust standard errors for panel regressions with cross-sectional dependence". *The Stata Journal*, 7(3), 281-312.
- Vogelsang, T.J. (2012). "Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects". *Journal of Econometrics*, 166(2), 303-319.
