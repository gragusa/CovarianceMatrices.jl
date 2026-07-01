# Heteroskedasticity-Robust (HC/HR) Estimators

HC (Heteroskedasticity-Consistent) estimators, also known as HR (Heteroskedasticity-Robust) estimators, provide robust covariance matrices for cross-sectional data with conditional heteroskedasticity.

## Mathematical Foundation

For cross-sectional data where $\mathbb{E}[\varepsilon_t^2 \mid X_t] = \sigma_t^2$ (heteroskedasticity) but $\mathbb{E}[\varepsilon_t \varepsilon_s \mid X_t, X_s] = 0$ for $t \neq s$ (no serial correlation), the robust covariance matrix is

```math
\hat{\Omega}_{HC} = \frac{1}{T} \sum_{t=1}^T \phi_j(h_t)\, g_t g_t',
```

where $g_t$ is the moment contribution of observation $t$ and $\phi_j(h_t)$ is an adjustment factor that depends on the leverage $h_t$ and the HC variant $j$.

## Available Estimators

```@docs
HC0
HC1
HC2
HC3
HC4
HC4m
HC5
```

The variants differ only in $\phi_j$. Writing $\bar h = k/n$ for the average leverage,

| Variant | $\phi_j(h_t)$ |
|---------|---------------|
| HC0 | $1$ |
| HC1 | $n/(n-k)$ |
| HC2 | $1/(1-h_t)$ |
| HC3 | $1/(1-h_t)^2$ |
| HC4 | $1/(1-h_t)^{\delta_t}$, $\delta_t = \min(4,\, h_t/\bar h)$ |
| HC4m | $1/(1-h_t)^{\delta_t}$, $\delta_t = \min(1,\, h_t/\bar h) + \min(1.5,\, h_t/\bar h)$ |
| HC5 | $1/(1-h_t)^{\alpha_t/2}$, $\alpha_t = \min\!\big(h_t/\bar h,\ \max(4,\, 0.7\, h_{\max}/\bar h)\big)$ |

HC0 is White's original estimator. HC1 applies a degrees-of-freedom correction. The remaining variants use the leverages $h_t$, the diagonal of the hat matrix $H = X(X'X)^{-1}X'$, to inflate the contribution of high-leverage observations.

## Usage

The estimators attach to a fitted GLM model through `vcov` and `stderror`. The California test-score data regresses average test scores on the student–teacher ratio and district income, a setting where the error variance falls with income.

```@example hc
using CovarianceMatrices, GLM, RDatasets

caschool = RDatasets.dataset("Ecdat", "Caschool")
model = lm(@formula(TestScr ~ Str + AvgInc), caschool)
stderror(model)            # classical (homoskedastic) standard errors
```

Passing an HC estimator as the first argument returns the robust standard errors:

```@example hc
stderror(HC3(), model)
```

The full robust covariance matrix is available through `vcov`:

```@example hc
vcov(HC3(), model)
```

The variants give the same point estimates but differ in their finite-sample correction:

```@example hc
using DataFrames
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    HC0 = stderror(HC0(), model),
    HC1 = stderror(HC1(), model),
    HC3 = stderror(HC3(), model),
)
```

## Alternative Names (HR)

The same estimators are available under the HR (Heteroskedasticity-Robust) names. `HRk` is an alias for `HCk`:

```@docs
HR0
HR1
HR2
HR3
HR4
HR4m
HR5
```

```@example hc
vcov(HC3(), model) == vcov(HR3(), model)
```

## References

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". *Econometrica*, 48(4), 817-838.
- MacKinnon, J.G. and White, H. (1985). "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties". *Journal of Econometrics*, 29(3), 305-325.
- Cribari-Neto, F. (2004). "Asymptotic inference under heteroskedasticity of unknown form". *Computational Statistics & Data Analysis*, 45(2), 215-233.
