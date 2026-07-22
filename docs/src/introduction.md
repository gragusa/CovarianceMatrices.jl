# Introduction and Mathematical Foundation

```@meta
CurrentModule = CovarianceMatrices
```

This page develops the theory behind [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/)
from the ground up and shows how each mathematical object maps onto a function
call. It is organised around two problems that look different but share a single
computational core.

**Part I — the sample mean.** We observe a sequence $\{X_t\}$ and want the
asymptotic variance of its average $\bar{X}_T$. When the observations are
dependent — serially (a time series) or in blocks (clusters) — that variance is
a *long-run variance*, not a plain covariance. The workhorse
[`aVar`](@ref) estimates it. This part fixes the estimand, walks through the
uncorrelated, time-correlated, and cluster-correlated cases, and pins down the
`scale` keyword precisely, because its meaning is the source of most confusion.

**Part II — an estimated parameter.** Most estimators are not sample means; they
solve an estimating equation $\frac{1}{T}\sum_t g(z_t,\hat\theta)=0$. We show that
the variance of $\hat\theta$ is a *sandwich* whose filling is exactly the
long-run variance of Part I, now applied to the moment (score) vectors. The
linear model is the leading special case. We then explain how the GLM.jl
extension supplies that filling automatically, and how you can teach the package
about a new estimator of your own.

Throughout, code blocks marked `@example` are executed when the documentation is
built, so every printed number below is produced by the package itself.

---

# Part I — The asymptotic variance of a sample mean

## The estimand

Let $\{X_t : 1 \le t \le T\}$ be a sequence of $k$-dimensional random vectors with
common mean $\mu = \mathbb{E}(X_t)$, and let

```math
\bar{X}_T = \frac{1}{T}\sum_{t=1}^{T} X_t
```

be the sample mean. Under regularity conditions (stationarity or a suitable
heterogeneity condition, plus a mixing/weak-dependence restriction and moment
bounds — see Newey and West, 1987; Andrews, 1991), $\bar{X}_T$ is asymptotically
normal:

```math
\sqrt{T}\,\Sigma^{-1/2}\left(\bar{X}_T - \mu\right) \xrightarrow{d} N(0, I_k),
```

where $\Sigma$ is the **asymptotic (long-run) variance** of the normalised sum,

```math
\Sigma
= \operatorname{a\mathbb{V}ar}\!\left(\frac{1}{\sqrt{T}}\sum_{t=1}^{T} X_t\right)
= \lim_{T\to\infty}\operatorname{Var}\!\left(\frac{1}{\sqrt{T}}\sum_{t=1}^{T} X_t\right).
```

Expanding the variance of the sum and collecting terms by the lag $\tau = t-s$
gives the fundamental identity

```math
\Sigma
= \underbrace{\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\!\left(X_t-\mu\right)\left(X_t-\mu\right)'}_{\text{contemporaneous variance}}
+ \underbrace{\frac{1}{T}\sum_{\tau=1}^{T-1}\sum_{t=\tau+1}^{T}\left[\mathbb{E}\!\left(X_t-\mu\right)\left(X_{t-\tau}-\mu\right)' + \mathbb{E}\!\left(X_{t-\tau}-\mu\right)\left(X_t-\mu\right)'\right]}_{\text{contribution of all autocovariances}} .
```

The asymptotic variance is therefore the average own-variance of $X_t$ **plus** a
term that accumulates every cross-covariance between $X_t$ and its lags. Writing
the $\tau$-th autocovariance as
$\Gamma_\tau = \mathbb{E}\!\left(X_t-\mu\right)\left(X_{t-\tau}-\mu\right)'$, the
stationary version collapses to the compact form

```math
\Sigma = \Gamma_0 + \sum_{\tau=1}^{\infty}\left(\Gamma_\tau + \Gamma_\tau'\right) = \sum_{\tau=-\infty}^{\infty} \Gamma_\tau ,
```

which is $2\pi$ times the spectral density of $\{X_t\}$ at frequency zero. This
single object — the **spectral density at frequency zero** — is what every
estimator in the package targets. The methods differ only in how they weight and
truncate the sample autocovariances $\hat\Gamma_\tau$ that estimate the
$\Gamma_\tau$. The special cases below correspond to assumptions that switch off
or restructure parts of the double sum.

## The `aVar` function and the `scale` convention

The single entry point for Part I is

```julia
aVar(estimator, X; demean=true, dims=1, means=nothing, prewhite=false, scale=true)
```

Here `X` is a $T \times k$ data matrix (observations in rows when `dims=1`), and
`estimator` selects how the autocovariances are weighted. Every concrete
estimator is a subtype of [`AbstractAsymptoticVarianceEstimator`](@ref); the ones
that model dependence ([`HAC`](estimators/hac.md), [`Cluster`](@ref)/[`CR`](@ref CR0),
[`EWC`](@ref), [`VARHAC`](@ref), [`DriscollKraay`](@ref)) share the abstract supertype
[`Correlated`](@ref), while the heteroskedasticity-only [HC/HR family](estimators/hc.md)
([`HC0`](@ref)–[`HC5`](@ref)) targets the contemporaneous term alone.

The keywords are:

  - `demean` — subtract the column means before forming autocovariances (the
    $X_t - \mu$ in the formulas). Set `false` if `X` is already centred (for
    example, it holds residuals or scores that sum to zero).
  - `dims` — `1` if observations are rows, `2` if columns.
  - `means` — supply known means instead of estimating them.
  - `prewhite` — for [`HAC`](estimators/hac.md) only: fit a VAR(1), apply the kernel to its
    residuals, then recolour (Andrews and Monahan, 1992).
  - `scale` — the normalisation, explained next.

### What `scale` does

Internally the estimator first forms an **unnormalised** matrix

```math
\hat{S} = \sum_{t} \hat{\Gamma}_0 \text{-type terms}
        = \widehat{\operatorname{Var}}\!\left(\sum_{t=1}^{T} X_t\right)
```

— the weighted double sum of Part I written *without* the leading $1/T$ — and then
divides it according to `scale`:

| `scale` | returned matrix | interpretation |
|---|---|---|
| `true` (default) | $\hat\Sigma = \hat S / T$ | estimate of $\operatorname{a\mathbb{V}ar}\!\big(T^{-1/2}\sum_t X_t\big)$, i.e. the variance of $\sqrt{T}(\bar X_T-\mu)$ |
| `false` | $\hat S$ | the unnormalised "meat" $\widehat{\operatorname{Var}}(\sum_t X_t)$ used by the sandwich formulas of Part II |
| `m::Int` | $\hat S / m$ | divide by `m` instead of `T`; use `m = T-k` for a degrees-of-freedom correction |

The default `scale=true` returns $\hat\Sigma$, the estimate of the $\Sigma$ in the
central limit theorem above. Two consequences are worth stating explicitly,
because they are the two most common mistakes:

!!! note "From `aVar` to the variance of the mean"
    `aVar(estimator, X)` estimates the variance of the **normalised** sum
    $\sqrt{T}(\bar X_T-\mu)$, *not* the variance of $\bar X_T$ itself. To obtain
    the (finite-sample) variance of the mean, divide once more by $T$:
    ```math
    \widehat{\operatorname{Var}}(\bar X_T) = \frac{\hat\Sigma}{T} = \frac{1}{T}\,\texttt{aVar(estimator, X)} .
    ```

!!! tip "`scale=false` is for building sandwiches"
    Pass `scale=false` when the result feeds a sandwich variance (Part II). There
    the "bread" already carries the compensating factors of $T$, so the meat must
    be the raw sum $\hat S$. This is exactly what the model-level `vcov` methods
    request internally.

### Seeing the convention in numbers

For the uncorrelated case the long-run variance is just the contemporaneous
variance $\Gamma_0$, and the estimator that targets it at the matrix level is
[`HC0`](@ref) (equivalently [`HR0`](@ref); the heteroskedasticity-robust "meat").
With centred data $\hat S = \tilde X'\tilde X$ (the sum of outer products), and the
three `scale` settings reproduce familiar quantities:

```@example scale
using CovarianceMatrices, Statistics, Random, LinearAlgebra
Random.seed!(1234)

T, k = 500, 3
X = randn(T, k) .* [1.0 2.0 0.5] .+ [0.3 -1.0 4.0]  # heteroskedastic columns, nonzero means

Σ̂    = aVar(HC0(), X)                 # scale=true  (default); HC0 = uncorrelated meat
Ŝ    = aVar(HC0(), X; scale=false)    # unnormalised sum
Σ̂dof = aVar(HC0(), X; scale=T-1)      # divide by T-1 instead of T

(; scale_true_over_cov = Σ̂ ./ cov(X; corrected=false),   # ≈ 1: matches the ÷T sample covariance
   sum_over_T           = (Ŝ ./ T) ./ Σ̂,                 # ≈ 1: scale=false is T × scale=true
   dof_over_cov         = Σ̂dof ./ cov(X))                # ≈ 1: scale=T-1 matches the ÷(T-1) covariance
```

All three ratio matrices are ones up to floating-point error, confirming that
`scale=true` gives the (uncorrected) sample covariance, `scale=false` gives its
$T$-fold sum, and `scale=T-1` reproduces the degrees-of-freedom-corrected
covariance. The standard error of each mean then divides by another $\sqrt{T}$:

```@example scale
se_mean      = sqrt.(diag(aVar(HC0(), X)) ./ T)  # √(Σ̂/T)
se_mean_naive = vec(std(X; dims=1) ./ sqrt(T))            # textbook s/√T, uncorrelated case
(; se_mean, se_mean_naive)
```

Under independence the two agree; the value of `aVar` is that it stays correct
when independence fails, which is the subject of the next three cases.

## Case 1 — Uncorrelated observations

When $\{X_t\}$ is serially uncorrelated,
$\Gamma_\tau = 0$ for all $\tau \ne 0$, and the double sum vanishes:

```math
\Sigma = \Gamma_0 = \mathbb{E}\!\left(X_t-\mu\right)\left(X_t-\mu\right)' .
```

Conditional heteroskedasticity is still allowed — the columns of $X$ may have
different variances, as above. At the matrix level the estimator is [`HC0`](@ref)
(alias [`HR0`](@ref)), and (as just shown) $\hat\Sigma = \frac{1}{T}\sum_t \tilde X_t \tilde X_t'$.
When this object is attached to a regression it produces White's
heteroskedasticity-robust covariance matrix; the [HC/HR family](estimators/hc.md)
adds the leverage-based finite-sample corrections HC1–HC5.

## Case 2 — Time (serial) correlation

If $\{X_t\}$ is serially correlated, the autocovariances $\Gamma_\tau$ no longer
vanish and the full double sum is active. A useful intermediate case is
**finite** correlation: if $X_t$ follows a vector MA($q$) process
$X_t = \mu + \varepsilon_t + \Theta_1\varepsilon_{t-1} + \dots + \Theta_q\varepsilon_{t-q}$,
then $\Gamma_\tau = 0$ for $\tau > q$ and the sum truncates exactly at lag $q$:

```math
\Sigma = \Gamma_0 + \sum_{\tau=1}^{q}\left(\Gamma_\tau + \Gamma_\tau'\right) .
```

In general $q$ is unknown or infinite, and the sample autocovariances
$\hat\Gamma_\tau$ become too noisy to sum naively (their number grows with $T$).
The package offers four strategies, all consistent for the same $\Sigma$:

  - **Kernel HAC** — $\hat\Sigma = \hat\Gamma_0 + \sum_{\tau\ge 1} k(\tau/S_T)(\hat\Gamma_\tau + \hat\Gamma_\tau')$,
    downweighting distant lags with a [kernel](estimators/hac.md) $k(\cdot)$ and
    bandwidth $S_T$ chosen by the [`Andrews`](@ref) or [`NeweyWest`](@ref) rules.
  - **[VARHAC](estimators/varhac.md)** — fit a
    VAR($p$) with order chosen by an information criterion and read $\Sigma$ off
    the implied spectral density; no kernel or bandwidth.
  - **[Smoothed moments](estimators/smoothed_moments.md)** — smooth the series
    before taking outer products; automatically positive semidefinite.
  - **[EWC](estimators/ewc.md)** — a low-rank
    cosine-basis estimate of the zero-frequency spectrum.

Mapping onto `aVar` is direct. The following simulates a stationary VAR(1),
$X_t = \Phi X_{t-1} + \varepsilon_t$, whose long-run variance genuinely exceeds
$\Gamma_0$, and compares the naive uncorrelated estimate against several
serial-correlation-robust ones:

```@example serial
using CovarianceMatrices, LinearAlgebra, Random
Random.seed!(42)

T, k = 600, 2
Φ = [0.7 0.1; 0.0 0.5]
X = zeros(T, k)
for t in 2:T
    X[t, :] = Φ * X[t-1, :] + randn(k)
end

estimators = ["HC0 (ignores dependence)"          => HC0(),
              "Bartlett{Andrews} (kernel HAC)"     => Bartlett{Andrews}(),
              "Parzen{NeweyWest} (kernel HAC)"     => Parzen{NeweyWest}(),
              "VARHAC (automatic lag order)"       => VARHAC(),
              "UniformSmoother"                    => UniformSmoother(round(Int, 2*T^(1/3))))

for (name, est) in estimators
    Σ̂ = aVar(est, X)
    println(rpad(name, 36), "  tr(Σ̂) = ", round(tr(Σ̂), digits=3))
end
```

The `HC0` estimator understates the long-run variance because it discards
the positive autocovariances that the AR dynamics inject; the HAC, VARHAC, and
smoothed estimators recover them.

## Case 3 — Cluster correlation

A related but distinct dependence structure arises when observations are
correlated **within groups** but independent **across groups** — firms observed
over several years, students within schools, repeated measurements on the same
subject. Index the groups by $g \in \{1,\dots,G\}$ and let each observation belong
to exactly one group. Because cross-group covariances are zero, the double sum of
Part I keeps only within-group pairs, and it is cleanest to write $\Sigma$ in
terms of the **group sums** $u_g = \sum_{t \in g}(X_t-\mu)$:

```math
\Sigma = \frac{1}{T}\sum_{g=1}^{G} \mathbb{E}\!\left[u_g\,u_g'\right]
       = \frac{1}{T}\sum_{g=1}^{G} \mathbb{E}\!\left[\Big(\sum_{t\in g}(X_t-\mu)\Big)\Big(\sum_{s\in g}(X_s-\mu)\Big)'\right].
```

Within a group the correlation pattern is left completely unrestricted (any
$\Gamma$ across the pair $t,s\in g$ is allowed); only the assumption that
different groups are independent is used. Estimators of $\Sigma$ replace the
expectation with the observed cluster sums,

```math
\hat\Sigma = \frac{1}{T}\,c\sum_{g=1}^{G}\hat u_g\,\hat u_g',
\qquad \hat u_g = \sum_{t\in g}\tilde X_t ,
```

with a small-sample factor $c$. For the sample-mean (matrix) case the estimator is
[`Cluster`](@ref), the clustering analogue of [`HC0`](@ref): it takes $c=1$, the
raw cluster long-run variance with no finite-sample correction. It carries the
group identifiers, so the same `aVar` call applies. Comparing the cluster-robust
estimate against the one that ignores grouping:

```@example cluster
using CovarianceMatrices, Random, LinearAlgebra
Random.seed!(7)

G, n_g, k = 40, 15, 2          # 40 clusters of 15 observations, 2 variables
clusters  = repeat(1:G, inner=n_g)
T = G * n_g

common = randn(G, k)[clusters, :]     # a shared shock per cluster => within-cluster correlation
X = common .+ randn(T, k)

println("HC0     (ignores clustering) : tr = ", round(tr(aVar(HC0(), X)), digits=3))
println("Cluster (cluster-robust)     : tr = ", round(tr(aVar(Cluster(clusters), X)), digits=3))
```

The shared per-cluster shock makes the true long-run variance much larger than
$\Gamma_0$; the `HC0` estimator misses it, while [`Cluster`](@ref) captures it. The
finite-sample variants [`CR0`](@ref)–[`CR3`](@ref) are reserved for the *regression*
setting: their degrees-of-freedom and leverage factors $c$ depend on the regressor
design matrix, so they attach to a fitted model rather than a bare matrix (Part II
and the [CR page](estimators/cr.md)).

## Case 4 — Panel data: correlation in two dimensions

Panel data can be dependent along **both** the time and the cross-sectional
dimension at once. The [`DriscollKraay`](@ref) estimator handles this by summing
each period's cross-section into a single vector and then applying a time-series
HAC kernel to the resulting sequence, so that arbitrary cross-sectional
correlation within a period and serial correlation across periods are both
absorbed:

```@example cluster
tis = repeat(1:30, inner=20)     # 30 time periods
iis = repeat(1:20, outer=30)     # 20 units
Xdk = randn(length(tis), 1) .+ randn(30)[tis]   # a common time shock across units
dk  = DriscollKraay(Bartlett{Andrews}(); tis=tis, iis=iis)
round(tr(aVar(dk, Xdk)), digits=3)
```

## Choosing an estimator (Part I)

| Dependence in $\{X_t\}$ | Estimator | Notes |
|---|---|---|
| None (heteroskedastic OK) | [`HC0`](@ref) / [`HR0`](@ref) | $\Sigma=\Gamma_0$ |
| Serial / time series | [`Bartlett`](@ref)`{`[`Andrews`](@ref)`}`, [`VARHAC`](@ref) | kernel or parametric |
| Serial, guaranteed PSD | [`UniformSmoother`](@ref), [`VARHAC`](@ref) | positive semidefinite by construction |
| Cluster / grouped | [`Cluster`](@ref) (matrix); [`CR1`](@ref)/[`CR2`](@ref)/[`CR3`](@ref) (model) | independent across groups |
| Panel (time + cross-section) | [`DriscollKraay`](@ref) | HAC over period sums |

With the long-run variance of a *sum* understood, we can turn to the object most
applications actually need: the variance of an *estimator*.

---

# Part II — Estimating equations with an estimated parameter

## From the sample mean to a general estimator

Few quantities of interest are literally sample means. Far more often a
$k$-dimensional parameter $\theta$ is defined as the solution of a system of
**moment conditions** (equivalently, first-order conditions, estimating
equations, or scores). Given data $\{z_t\}_{t=1}^{T}$ and a moment function
$g:\mathcal Z\times\Theta\to\mathbb R^{m}$ with $m\ge k$, the estimator
$\hat\theta$ satisfies, at least approximately,

```math
\frac{1}{T}\sum_{t=1}^{T} g(z_t,\hat\theta) = 0 .
```

Maximum likelihood ($g$ = the score), OLS and IV ($g$ = regressor $\times$
residual), and GMM (any set of orthogonality conditions) all fit this template.

A first-order (delta-method) expansion of the moment condition around the true
$\theta_0$ links the sampling error of $\hat\theta$ to the sampling error of the
*average moment*:

```math
\sqrt{T}\,(\hat\theta - \theta_0)
= -\,G^{-1}\,\frac{1}{\sqrt{T}}\sum_{t=1}^{T} g(z_t,\theta_0) + o_p(1),
\qquad
G = \mathbb{E}\!\left[\frac{\partial g(z_t,\theta_0)}{\partial \theta'}\right].
```

The right-hand side is a known matrix $G$ times a normalised sum — precisely the
object of Part I. Applying the central limit theorem to that sum, with its
long-run variance

```math
\Omega = \operatorname{a\mathbb{V}ar}\!\left(\frac{1}{\sqrt T}\sum_{t=1}^{T} g(z_t,\theta_0)\right)
```

(the *same* spectral-density-at-zero quantity as before, now for the moment
vectors), the estimator is asymptotically normal with a **sandwich** variance:

```math
\sqrt{T}\,(\hat\theta-\theta_0)\xrightarrow{d} N\!\left(0,\; V\right),
\qquad
\boxed{\,V = G^{-1}\,\Omega\,G^{-\prime}\,}\quad(\text{just identified, } m=k).
```

This is the central message of Part II:

!!! note "Part II = Part I, wrapped in a Jacobian"
    The only new ingredient beyond Part I is the Jacobian $G$ (the "bread"). The
    filling $\Omega$ is the long-run variance of the estimated moment vectors
    $g(z_t,\hat\theta)$, computed by the very same [`aVar`](@ref) machinery — with
    `scale=false`, so that $\hat\Omega$ is the unnormalised sum the sandwich
    expects. Choosing an estimator for $\Omega$ (uncorrelated, HAC, cluster, …) is
    exactly the Part I decision, driven by the dependence in the *moments*.

The finite-sample covariance of $\hat\theta$ is then $\hat V/T$, mirroring the
"divide by another $T$" rule of Part I. In the package the factors of $T$ are
tracked internally, and the model-level `vcov` returns the covariance of
$\hat\theta$ directly.

## The linear model as the leading special case

Ordinary least squares is the sandwich with the simplest possible pieces. With
regressors $x_t\in\mathbb R^{k}$, residuals $\hat u_t = y_t - x_t'\hat\beta$, and
moment function $g(z_t,\beta) = x_t\,(y_t - x_t'\beta)$, the moment matrix stacks
the rows $g_t' = \hat u_t\, x_t'$, and the Jacobian is

```math
G = \mathbb{E}\!\left[\frac{\partial g_t}{\partial\beta'}\right] = -\,\mathbb{E}[x_t x_t'],
\qquad
\hat G = -\frac{1}{T}X'X .
```

Substituting into the sandwich, the factors of $T$ cancel and we recover the
textbook robust covariance matrix

```math
\hat V_{\hat\beta}
= (X'X)^{-1}\Big(\underbrace{\textstyle\sum_t \text{weighted } \hat u_t^2\, x_t x_t'}_{\hat\Omega\ =\ \texttt{aVar(est, g; scale=false)}}\Big)(X'X)^{-1}.
```

The "bread" $(X'X)^{-1}$ is $\hat G^{-1}$ up to the factor of $T$ that the
`scale=false` meat carries — the two `scale` conventions of Part I are exactly
what make this cancellation clean. Every choice of $\hat\Omega$ reproduces a named
estimator: the [HC family](estimators/hc.md) gives White's heteroskedasticity-robust
matrix, a [`HAC`](estimators/hac.md) kernel gives Newey–West, and [`CR1`](@ref) gives
cluster-robust standard errors. Nonlinear least squares, IV, and GLMs replace
$x_t$ and $\hat u_t$ with the appropriate score, but the structure is identical.

## Correct specification versus robustness: the two variance forms

When $m=k$ (exactly identified, as in MLE) the sandwich $V=G^{-1}\Omega G^{-\prime}$
can be simplified *if the model is correctly specified*. The information-matrix
equality then gives $\Omega = -G$ (Fisher information), and the sandwich collapses
to $V = -G^{-1} = \mathcal I^{-1}$, the Cramér–Rao bound. Dropping that assumption
keeps the full sandwich, which stays valid under misspecification. The package
exposes this choice through two [`VarianceForm`](@ref) singletons:

| | [`Information`](@ref) (correct spec.) | [`Misspecified`](@ref) (robust) |
|---|---|---|
| [`MLikeModel`](@ref) ($m=k$) | $V = H^{-1}$ | $V = H^{-1}\,\Omega\,H^{-1}$ |
| [`GMMLikeModel`](@ref) ($m\ge k$) | $V = (G'\Omega^{-1}G)^{-1}$ | $V=(G'WG)^{-1}G'W\Omega W G\,(G'WG)^{-1}$ |

Here $H = \sum_t \partial g_t/\partial\theta'$ is the Hessian of the objective
([`hessian_objective`](@ref)), $G$ the moment Jacobian
([`jacobian_momentfunction`](@ref)), $\Omega$ the long-run variance of the moments
([`aVar`](@ref) with `scale=false`), and $W$ an optional GMM weight matrix
([`weight_matrix`](@ref)). The efficient GMM choice $W=\Omega^{-1}$ yields the
`Information` row; the common $W=I$ "two-step" estimator uses the `Misspecified`
row. These formulas are assembled by

```julia
vcov(estimator, form, model)       # form :: Information() or Misspecified()
stderror(estimator, form, model)
```

## The GLM.jl extension

For fitted GLM.jl models you do not implement anything: loading GLM.jl activates
an extension that supplies the sandwich ingredients from the model's internals,
so the HC, HAC, and CR estimators all attach directly. The extension
maps the abstract interface onto GLM's fitted objects roughly as follows:

| Interface object | Linear model | Generalized linear model |
|---|---|---|
| [`momentmatrix`](@ref) $g_t$ | $x_t\,\hat u_t$ | $x_t\, w_t\, \hat u_t^{\text{wrk}}/\phi$ (working residuals) |
| `bread` $\hat G^{-1}$ | $(X'X)^{-1}$ | $(X'WX)^{-1}\phi$ |
| `leverage` $h_t$ | hat-matrix diagonal | weighted hat-matrix diagonal |

where $w_t$ are the IRLS working weights and $\phi$ the dispersion parameter. With
those in place, robust standard errors are one call. The monthly US `economics`
series regresses the personal saving rate on unemployment, whose residuals are
strongly serially correlated, so classical and HAC standard errors diverge:

```@example glm
using CovarianceMatrices, GLM, RDatasets, DataFrames

econ  = RDatasets.dataset("ggplot2", "economics")
model = lm(@formula(PSavert ~ Unemploy), econ)

DataFrame(
    coefficient = coefnames(model),
    classical   = stderror(model),                       # assumes i.i.d. errors
    hc1         = stderror(HC1(), model),                # heteroskedasticity-robust
    hac_bart    = stderror(Bartlett{Andrews}(), model),  # serial-correlation-robust
    hac_parzen  = stderror(Parzen{Andrews}(), model),    # different kernel/bandwidth rule
)
```

The same calls work on `glm(...)` fits (logit, Poisson, …) — the working-residual
row of the table above is what makes the linear formulas carry over unchanged.
The `Information`/`Misspecified` distinction is also available on these models via
`vcov(estimator, Information(), model)` when you want the correctly-specified form
rather than the robust sandwich.

## Extending the package to a custom estimator

Any estimator that can produce a moment matrix can obtain robust standard errors,
whether or not it is a GLM. The package uses **duck typing**: implement a few
methods and every estimator becomes available. The required methods are
[`momentmatrix`](@ref), `StatsBase.coef`, and `StatsBase.nobs`; a
[`GMMLikeModel`](@ref) additionally needs [`jacobian_momentfunction`](@ref) (the
$G$ matrix), and the `Misspecified` form needs [`hessian_objective`](@ref).

As a self-contained worked example, consider an **instrumental variables**
estimator — the natural generalisation of the linear model, in which each
regressor is replaced by an instrument $z_t$ (OLS is the special case $Z=X$). We
take the just-identified case $m=k$, where the robust variance is the clean
sandwich $\hat G^{-1}\hat\Omega\,\hat G^{-\prime}$; over-identification ($m>k$) is
handled by the same interface with an extra [`weight_matrix`](@ref) and
[`hessian_objective`](@ref), as shown in the tutorial. The moment conditions and
their Jacobian are

```math
g(z_t,\beta) = z_t\,(y_t - x_t'\beta),
\qquad
G = \mathbb{E}\!\left[\frac{\partial g_t}{\partial\beta'}\right] = -\,\mathbb{E}[z_t x_t'],
\qquad
\hat G = -\,Z'X ,
```

which we can write in closed form (no optimiser needed). Declaring the type a
subtype of [`GMMLikeModel`](@ref) and implementing four short methods is enough:

```@example extend
using CovarianceMatrices, LinearAlgebra, Random, StatsBase

# --- a minimal over-identified IV/GMM estimator -------------------------------
struct IVGMM <: CovarianceMatrices.GMMLikeModel
    y::Vector{Float64}
    X::Matrix{Float64}   # T×k regressors (some endogenous)
    Z::Matrix{Float64}   # T×m instruments, here m = k
    β::Vector{Float64}   # fitted coefficients
end

IVGMM(y, X, Z) = IVGMM(y, X, Z, (Z'X) \ (Z'y))   # just-identified IV estimator

# --- the interface: four methods ---------------------------------------------
CovarianceMatrices.momentmatrix(m::IVGMM)         = (m.y .- m.X * m.β) .* m.Z   # T×m
CovarianceMatrices.jacobian_momentfunction(m::IVGMM) = -(m.Z' * m.X)            # m×k  (= Ĝ)
StatsBase.coef(m::IVGMM) = m.β
StatsBase.nobs(m::IVGMM) = length(m.y)
nothing # hide
```

Simulate an endogenous regressor with valid instruments and fit:

```@example extend
Random.seed!(2024)
T = 500
u = randn(T)                           # structural error
z = randn(T)                           # instrument (valid: independent of u)
x = z .+ 0.7u                          # endogenous regressor (correlated with u)
X = [ones(T)  x]                       # k = 2 regressors  (intercept + x)
Z = [ones(T)  z]                       # m = 2 instruments (intercept + z)
β0 = [1.0, 2.0]
y = X * β0 .+ u

model = IVGMM(y, X, Z)
round.(coef(model), digits=3)          # ≈ β0; OLS would be biased by the endogeneity
```

Because `IVGMM` is a `GMMLikeModel`, the full sandwich is assembled automatically;
choosing the dependence structure of $\Omega$ is exactly the Part I decision. With
`HC0` the moments are treated as uncorrelated, and a HAC filling drops straight in
for the case where they are serially correlated:

```@example extend
se_iv  = stderror(HC0(),               Information(), model)  # robust IV sandwich, uncorrelated moments
se_hac = stderror(Bartlett{Andrews}(), Information(), model)  # if the moments were serially correlated
(; se_iv, se_hac)
```

The [Package Interface Tutorial](tutorials/interface_tutorial.md) works through
further cases — an M-estimator, a maximum-likelihood model with both variance
forms, and a model that skips the abstract supertype entirely.

## Summary

Both halves of the package rest on one estimand, the long-run variance
$\Omega=\sum_\tau\Gamma_\tau$. Part I estimates it for a raw data matrix through
[`aVar`](@ref), with the `scale` keyword selecting whether the result is the
variance of the normalised sum (`true`), the unnormalised meat for a sandwich
(`false`), or a degrees-of-freedom-corrected variant (`m::Int`). Part II wraps
that same $\Omega$ in a Jacobian to produce the variance of an estimated
parameter; the linear model is the special case with bread $(X'X)^{-1}$, GLM.jl
fits get their ingredients from an extension, and any custom estimator joins by
implementing a handful of interface methods.

## Where to go next

  - [HAC Estimators](estimators/hac.md), [HC/HR](estimators/hc.md),
    [CR](estimators/cr.md), [VARHAC](estimators/varhac.md),
    [Smoothed Moments](estimators/smoothed_moments.md), [Driscoll-Kraay](estimators/driscoll_kraay.md),
    and [EWC](estimators/ewc.md) — one page per estimator.
  - [Matrix Interface Tutorial](tutorials/matrix_tutorial.md) — more `aVar` recipes (Part I).
  - [GLM Integration Tutorial](tutorials/glm_tutorial.md) — robust inference with GLM.jl (Part II).
  - [Package Interface Tutorial](tutorials/interface_tutorial.md) — extend the package to custom models.
  - [API Reference](@ref) — every exported function and type.

## References

  - Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica*, 59(3), 817–858.
  - Andrews, D.W.K. and Monahan, J.C. (1992). "An Improved Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimator." *Econometrica*, 60(4), 953–966.
  - Cameron, A.C. and Miller, D.L. (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources*, 50(2), 317–372.
  - den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation." *Handbook of Statistics*, 15, 291–341.
  - Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data." *Review of Economics and Statistics*, 80(4), 549–560.
  - Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029–1054.
  - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.
  - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817–838.
