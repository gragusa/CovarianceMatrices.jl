# GLM Integration Tutorial

This tutorial shows how to use CovarianceMatrices.jl with GLM.jl for robust
inference in regression models.

## Overview

CovarianceMatrices.jl adds methods to GLM.jl's `vcov` and `stderror`, so a robust
covariance estimator is supplied as the first argument:

1. Fit the model with GLM.jl (`lm`, `glm`)
2. Compute the robust covariance with `vcov(estimator, model)`
3. Take robust standard errors with `stderror(estimator, model)`

```@example glm
using CovarianceMatrices, GLM, RDatasets, DataFrames, StatsBase, LinearAlgebra, Statistics
```

## Example 1: Time Series Regression

The `Capm` data hold 516 monthly returns. Regressing the food-industry excess
return on the market excess return gives a CAPM market model whose residuals are
both heteroskedastic and serially correlated.

```@example glm
capm = dataset("Ecdat", "Capm")
capm.ExFood = capm.RFood .- capm.RF

model = lm(@formula(ExFood ~ RMRF), capm)
```

### Heteroskedasticity-Robust Standard Errors

The HC estimators relax the constant-variance assumption. `HC1` applies the
`n/(n-k)` correction; `HC2` and `HC3` downweight high-leverage observations and are
the usual choice in samples of moderate size.

```@example glm
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    HC0 = stderror(HC0(), model),
    HC1 = stderror(HC1(), model),
    HC2 = stderror(HC2(), model),
    HC3 = stderror(HC3(), model),
)
```

The robust standard error on the slope is about 37% larger than the classical one,
so the classical figure overstates the precision of the market beta. The intercept
is barely affected.

### HAC Standard Errors

Monthly returns are also serially correlated, which the HC estimators do not
address. HAC estimators correct for both:

```@example glm
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    bartlett = stderror(Bartlett{Andrews}(), model),
    parzen = stderror(Parzen{NeweyWest}(), model),
    quadratic = stderror(QuadraticSpectral{Andrews}(), model),
    varhac = stderror(VARHAC(), model),
    smoothed = stderror(UniformSmoother(round(Int, 2.0 * nobs(model)^(1 / 3))), model),
)
```

Accounting for serial correlation widens the standard errors further. `VARHAC` and
the smoothed-moments estimators reach a similar answer without a bandwidth choice.

### Bandwidth and Lag Diagnostics

`optimalbw` reports the bandwidth a data-driven rule selects for a moment matrix.
`bandwidth` and `order` read what was actually used off the returned
`CovarianceMatrix`:

```@example glm
V_bartlett = vcov(Bartlett{Andrews}(), model)
V_varhac = vcov(VARHAC(), model)

(bartlett_bandwidth = bandwidth(V_bartlett)[1],
 varhac_lags = CovarianceMatrices.order(V_varhac))
```

## Example 2: Panel Data and Clustered Standard Errors

The Grunfeld data follow ten firms over twenty years. Investment is persistent
within a firm, so the independence assumption behind the classical and HC standard
errors fails.

```@example glm
grunfeld = dataset("plm", "Grunfeld")
panel = lm(@formula(Inv ~ Value + Capital), grunfeld)

DataFrame(
    coef = coefnames(panel),
    classical = stderror(panel),
    HC1 = stderror(HC1(), panel),
    CR0_firm = stderror(CR0(grunfeld.Firm), panel),
    CR1_firm = stderror(CR1(grunfeld.Firm), panel),
    CR2_firm = stderror(CR2(grunfeld.Firm), panel),
    CR3_firm = stderror(CR3(grunfeld.Firm), panel),
)
```

Clustering by firm roughly doubles the standard errors relative to `HC1`. `CR0`
applies no small-sample correction and `CR3` the heaviest, a spread that matters
here because there are only ten clusters.

```@example glm
CovarianceMatrices.nclusters(CR1(grunfeld.Firm))
```

Two-way clustering and Driscoll-Kraay allow for dependence across firms within a
year as well:

```@example glm
dk = DriscollKraay(Bartlett{Andrews}(), tis = grunfeld.Year, iis = grunfeld.Firm)

DataFrame(
    coef = coefnames(panel),
    CR1_firm = stderror(CR1(grunfeld.Firm), panel),
    CR1_twoway = stderror(CR1((grunfeld.Firm, grunfeld.Year)), panel),
    driscoll_kraay = stderror(dk, panel),
)
```

## Example 3: Logistic Regression

The estimators apply to generalized linear models as well. The Boston HMDA data
record whether a mortgage application was denied, along with the debt-to-income
ratio and the loan-to-value ratio.

```@example glm
hmda = dropmissing(dataset("Ecdat", "Hdma"), [:Deny, :DIR, :LVR, :Black])
hmda.denied = hmda.Deny .== "yes"
hmda.black = hmda.Black .== "yes"

logit = glm(@formula(denied ~ DIR + LVR + black), hmda, Binomial(), LogitLink())
```

```@example glm
DataFrame(
    coef = coefnames(logit),
    estimate = coef(logit),
    classical = stderror(logit),
    HC0 = stderror(HC0(), logit),
    HC3 = stderror(HC3(), logit),
    CR1_credit = stderror(CR1(hmda.CCS), logit),
)
```

Clustering here is on the credit score category, `CCS`.

## Inference

`vcov` and `stderror` are the two methods that take an estimator. Build the test
statistics from the robust standard errors:

```@example glm
se = stderror(HC3(), logit)
estimates = coef(logit)

DataFrame(
    coef = coefnames(logit),
    estimate = estimates,
    se = se,
    z = estimates ./ se,
    lower = estimates .- 1.96 .* se,
    upper = estimates .+ 1.96 .* se,
)
```

## Choosing an Estimator

| Data structure | Estimator |
|---|---|
| Cross-section, moderate sample | `HC2` or `HC3` |
| Cross-section, large sample | `HC0` or `HC1` |
| Time series | `Bartlett{Andrews}`, or `VARHAC` to avoid choosing a bandwidth |
| Clustered | `CR1(g)`, or `CR1((g1, g2))` for two-way |
| Panel with cross-sectional dependence | `DriscollKraay` |

Two cautions. HAC estimates depend on the bandwidth, so report the one you used;
`bandwidth` retrieves it. The CR estimators rely on a growing number of clusters,
and with few clusters — ten in the Grunfeld example — they understate uncertainty
whichever correction is applied.

The [Matrix Interface Tutorial](matrix_tutorial.md) covers the same estimators
applied directly to a moment matrix, and the
[Package Interface Extension](interface_tutorial.md) shows how to support a custom
model type.
