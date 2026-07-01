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
\qquad \hat{\Gamma}_j = \frac{1}{T}\sum_{t=j+1}^{T} h_t h_{t-j}'.
```

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

For a `RegressionModel`, `vcov(::DriscollKraay, model)` accepts a `type` argument that selects a finite-sample correction, reproducing the scalar `type` options of R's `plm::vcovSCC`. With `n` observations, `k` coefficients, and `T` time periods:

| `type` | Factor $c$ | Description |
|--------|------------|-------------|
| `:HC0` (default) | $1$ | No correction |
| `:HC1` | $n/(n-k)$ | Degrees-of-freedom correction |
| `:sss` | $\dfrac{n-1}{n-k}\cdot\dfrac{T}{T-1}$ | Stata-style small-sample correction |

```@example dk
fixed = DriscollKraay(Bartlett(5), tis=grunfeld.Year, iis=grunfeld.Firm)
DataFrame(
    coef = coefnames(model),
    HC0 = sqrt.(diag(vcov(fixed, model))),
    HC1 = sqrt.(diag(vcov(fixed, model; type=:HC1))),
    sss = sqrt.(diag(vcov(fixed, model; type=:sss))),
)
```

A bandwidth of `Bartlett(b)` corresponds to `plm::vcovSCC(..., maxlag = b - 1)`. The leverage-based `:HC2`/`:HC3`/`:HC4` options of `plm::vcovSCC` are not supported.

## References

- Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data". *Review of Economics and Statistics*, 80(4), 549-560.
- Hoechle, D. (2007). "Robust standard errors for panel regressions with cross-sectional dependence". *The Stata Journal*, 7(3), 281-312.
- Vogelsang, T.J. (2012). "Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects". *Journal of Econometrics*, 166(2), 303-319.
