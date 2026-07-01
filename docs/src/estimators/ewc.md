# Equal Weighted Cosine (EWC) Estimator

The Equal Weighted Cosine (EWC) estimator is a long-run variance estimator for serially correlated data. In place of the kernel and bandwidth of a HAC estimator, it projects the moments onto a small number of cosine basis functions and averages the resulting periodogram ordinates (Lazarus, Lewis, Stock, and Watson, 2018).

## Mathematical Foundation

For a moment series $g_1, \dots, g_T$, define the cosine projections

```math
\Lambda_j = \sqrt{2} \sum_{t=1}^{T} \cos\!\left(\frac{\pi j (t - 1/2)}{T}\right) g_t, \qquad j = 1, \dots, B.
```

The EWC estimator is the equally weighted average of their outer products,

```math
\hat{\Omega}_{EWC} = \frac{1}{B} \sum_{j=1}^{B} \Lambda_j \Lambda_j'.
```

The single tuning parameter $B$ is the number of basis functions. It plays the role the bandwidth plays in a HAC estimator: a smaller $B$ smooths more. The estimate is a sum of outer products, so it is positive semi-definite by construction, and the implied inference uses a $t$ (or $F$) reference distribution with degrees of freedom tied to $B$.

## Core Type

```@docs
EWC
```

## Usage

EWC attaches to a fitted model the same way the HAC estimators do. Using the monthly `economics` series, where the residuals of a saving-rate regression are serially correlated:

```@example ewc
using CovarianceMatrices, GLM, RDatasets

econ = RDatasets.dataset("ggplot2", "economics")
model = lm(@formula(PSavert ~ Unemploy), econ)
stderror(EWC(8), model)
```

The number of basis functions trades bias against variance. Few basis functions give a smoother, lower-variance estimate; more basis functions track the spectrum more closely at the cost of variance:

```@example ewc
using DataFrames
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    ewc4 = stderror(EWC(4), model),
    ewc8 = stderror(EWC(8), model),
    ewc16 = stderror(EWC(16), model),
)
```

## References

- Lazarus, E., Lewis, D.J., Stock, J.H., and Watson, M.W. (2018). "HAR Inference: Recommendations for Practice". *Journal of Business & Economic Statistics*, 36(4), 541-559.
- Müller, U.K. (2007). "A theory of robust long-run variance estimation". *Journal of Econometrics*, 141(2), 1331-1352.
