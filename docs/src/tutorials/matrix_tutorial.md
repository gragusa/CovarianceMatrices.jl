# Matrix Interface Tutorial

This tutorial shows how to use CovarianceMatrices.jl with the matrix interface for
direct covariance estimation. Use this approach when you have moment conditions or
residuals in hand and need a robust covariance matrix without fitting a model
through GLM.jl.

## Basic Workflow

The matrix interface follows this pattern:

1. Prepare your data matrix (moment conditions, residuals, etc.)
2. Choose an estimator
3. Compute the covariance matrix with `aVar`
4. Extract standard errors if needed

```@example matrix
using CovarianceMatrices, RDatasets, DataFrames, LinearAlgebra, Statistics
```

## Example 1: Time Series with Serial Correlation

The `Capm` data hold 516 monthly returns on three industry portfolios together with
the market factor and the risk-free rate. Monthly excess returns are serially
correlated, so they make a natural test case for HAC estimation.

```@example matrix
capm = dataset("Ecdat", "Capm")
X = Matrix(select(capm, [:RFood, :RDur, :RCon])) .- capm.RF
X = X .- mean(X, dims = 1)
size(X)
```

Each column is the demeaned excess return of one portfolio. `aVar` treats the rows
as observations and estimates the long-run covariance of the column means.

### HAC Estimation

HAC estimators account for both heteroskedasticity and autocorrelation:

```@example matrix
estimators = [
    "Bartlett (Andrews)" => Bartlett{Andrews}(),
    "Bartlett (fixed, 5)" => Bartlett(5),
    "Parzen (Newey-West)" => Parzen{NeweyWest}(),
    "Quadratic Spectral" => QuadraticSpectral{Andrews}(),
]

DataFrame(
    estimator = first.(estimators),
    trace = [tr(aVar(e, X)) for e in last.(estimators)],
)
```

### VARHAC Estimation

VARHAC fits a VAR to the moment matrix and reads the long-run covariance off the
fitted model, so no bandwidth is needed:

```@example matrix
V_aic = aVar(VARHAC(), X)        # defaults: AIC, SameLags(8)
V_bic = aVar(VARHAC(:bic), X)

DataFrame(
    selector = ["AIC", "BIC"],
    trace = [tr(V_aic), tr(V_bic)],
    lags = [CovarianceMatrices.order(V_aic), CovarianceMatrices.order(V_bic)],
)
```

`order` reads the selected lag length off the result, one entry per column of `X`.
BIC penalizes lags more heavily than AIC and here selects zero for every column, so
its estimate reduces to the contemporaneous covariance.

The accessors `order`, `AICs` and `BICs` take the `CovarianceMatrix` that `aVar`
returns, not the estimator, and are reached through the module since they are not
exported.

### Smoothed Moments Estimation

Smith's smoothed moments method yields a positive semi-definite estimate by
construction:

```@example matrix
T = size(X, 1)
smoothers = [
    "Uniform (rate rule)" => UniformSmoother(round(Int, 2.0 * T^(1 / 3))),
    "Triangular (rate rule)" => TriangularSmoother(round(Int, 1.5 * T^(1 / 5))),
    "Uniform (fixed, 8)" => UniformSmoother(8),
]

DataFrame(
    smoother = first.(smoothers),
    trace = [tr(aVar(s, X)) for s in last.(smoothers)],
)
```

## Example 2: Cross-Sectional Data with Heteroskedasticity

When observations are independent, the HC/HR estimators correct for
heteroskedasticity alone:

```@example matrix
resid = reshape(X[:, 1], :, 1)   # one column of moment contributions
aVar(HC0(), resid)[1, 1]
```

The corrections that distinguish `HC1` from `HC5` are functions of the design
matrix: `HC1` rescales by `n/(n-k)` and `HC2`–`HC5` use the leverage of each
observation. A bare moment matrix supplies neither, so every variant returns the
same number here. Apply them to a fitted model — see the
[GLM Integration Tutorial](glm_tutorial.md) — for the corrections to take effect.

## Example 3: Clustered Data

For a data or moment matrix whose observations are grouped, use the `Cluster`
estimator. The `CR0`–`CR3` estimators, with their degrees-of-freedom and leverage
corrections, are the *regression* interface — see the
[GLM Integration Tutorial](glm_tutorial.md). Those corrections need a fitted model's
design matrix; applied to a bare matrix the CR variants carry no correction and all
reduce to the raw cluster sum that `Cluster` returns.

The Grunfeld data track investment for ten firms over twenty years:

```@example matrix
grunfeld = dataset("plm", "Grunfeld")
inv_dev = reshape(grunfeld.Inv .- mean(grunfeld.Inv), :, 1)

DataFrame(
    estimator = ["HC0 (no clustering)", "Cluster by firm", "Cluster by year"],
    variance = [
        aVar(HC0(), inv_dev)[1, 1],
        aVar(Cluster(grunfeld.Firm), inv_dev)[1, 1],
        aVar(Cluster(grunfeld.Year), inv_dev)[1, 1],
    ],
)
```

Clustering by firm raises the variance substantially: investment is strongly
persistent within a firm, so the independent-observation estimate understates it.

## Example 4: Panel Data with Driscoll-Kraay

Driscoll-Kraay handles panels where units are correlated within a period and each
unit is correlated over time:

```@example matrix
dk = DriscollKraay(Bartlett{Andrews}(), tis = grunfeld.Year, iis = grunfeld.Firm)
aVar(dk, inv_dev)[1, 1]
```

## Example 5: EWC Estimation

The Equal Weighted Cosine estimator uses a number of basis functions in place of a
bandwidth:

```@example matrix
DataFrame(
    B = [5, 10, 15],
    trace = [tr(aVar(EWC(B), X)) for B in (5, 10, 15)],
)
```

## Prewhitening

Prewhitening fits a VAR(1) before applying the kernel, which can improve
finite-sample behavior when the series is persistent:

```@example matrix
DataFrame(
    prewhite = [false, true],
    trace = [
        tr(aVar(Bartlett{Andrews}(), X; prewhite = false)),
        tr(aVar(Bartlett{Andrews}(), X; prewhite = true)),
    ],
)
```

## Bandwidth Diagnostics

`optimalbw` returns the bandwidth a data-driven rule selects, and `bandwidth` reads
the bandwidth actually used off a result:

```@example matrix
bw_andrews = optimalbw(Bartlett{Andrews}(), X)
bw_newey = optimalbw(Bartlett{NeweyWest}(), X)
V = aVar(Bartlett{Andrews}(), X)

DataFrame(
    quantity = ["optimalbw (Andrews)", "optimalbw (Newey-West)", "bandwidth(V)"],
    value = [bw_andrews, bw_newey, bandwidth(V)[1]],
)
```

## Choosing an Estimator

| Data structure | Estimator |
|---|---|
| Cross-section, heteroskedasticity only | `HC0`–`HC5` |
| Time series, serial correlation | `Bartlett`, `Parzen`, `QuadraticSpectral` |
| Time series, no bandwidth choice | `VARHAC` |
| Time series, positive semi-definite by construction | `VARHAC`, `UniformSmoother`, `EWC` |
| Grouped observations | `Cluster`, or `CR0`–`CR3` for a fitted model |
| Panel with cross-sectional dependence | `DriscollKraay` |

HAC estimates are sensitive to the bandwidth, so report the one you used;
`bandwidth` retrieves it from the result. `VARHAC` and the smoothed-moments
estimators avoid the choice altogether. With few clusters the CR estimators are
unreliable regardless of the correction applied.

The [GLM Integration Tutorial](glm_tutorial.md) shows the same estimators applied to
a fitted model.
