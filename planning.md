This document describe a plan to improve the CovarianceMatrices.jl'API.

## Purpose

Provide a **singl interface** for third-party estimators to obtain asymptotic covariance matrices under either **correct specification** or **misspecification**.

The proposal is to work with the following two Abstract Type

* **MLikeModel (exactly identified, m = k)**

  * `:information` — correctly specified & iid
  * `:robust` — misspecified / heteroskedastic / dependent

* **GMMLikeModel (overidentified, m > k)**

  * `:correctly_specified` — moments correctly specified, optimal weight
  * `:misspecified` — robust to misspecification & dependence

These are subtype of StatsModel.StatisticalModel

## Some mathematics 

Let $g(x_i,\theta)\in\mathbb{R}^m$ and $\hat\theta$ solve $\frac{1}{n}\sum_i g(x_i,\hat\theta)=0$.

* `Z = momentmatrix(model)` → $n \times m$ with rows $g_i(\hat\theta)'$.
* `G = jacobian(model)` → $m \times k$, $\partial \bar g / \partial \theta'$ at $\hat\theta$.
* `H = objective_hessian(model)` → $k \times k$ (Hessian of the estimator’s objective at $\hat\theta$).

* For **MLikeModel (m=k)**: $H \equiv G$ (criterion Hessian equals Jacobian of the score).
* $\Omega$ is the long-run covariance of $\sqrt{n}\,\bar g$ (estimated from `Z` via the chosen `VarianceEstimator`).

---

## 2) Public API

### 2.1 High-level covariance

```julia
vcov(ve::VarianceEstimator, model;
     form::Symbol = :auto,        # :information | :robust | :correctly_specified | :misspecified | :auto
     W::Union{Nothing,AbstractMatrix}=nothing,  # optional W for GMM misspecified
     scale::Symbol = :n,          # Ω scaling, see §6
     rcond_tol::Real = 1e-12,
     check::Bool = true,
     warn::Bool = true) -> Matrix{Float64}

stderror(ve::VarianceEstimator, model; kwargs...) -> Vector{Float64}
```

**Form resolution when `form=:auto`:**

* Let `m = size(momentmatrix(model),2)` and `k = length(coef(model))`.
* If `m == k` (M-like): default **`:robust`** (safe). Users can set `form=:information`.
* If `m > k` (GMM-like): default **`:correctly_specified`** (your stated preference for GMM when moments are the focus and misspecification is “more problematic”). Users can set `form=:misspecified` explicitly to get the full robust GMM form.

I think it is best to use julia multiple dispatch

vcov(ve::VarianceEstimator, form::VarianceEstimatorType;
     form::Symbol = :auto,        # :information | :robust | :correctly_specified | :misspecified | :auto
     W::Union{Nothing,AbstractMatrix}=nothing,  # optional W for GMM misspecified
     scale::Symbol = :n,          # Ω scaling, see §6
     check::Bool = true,
     warn::Bool = true) -> Matrix{Float64}


where VarianceEstimatorType is abstract with subtype 
- Information 
- Robust 
- CorrectlySpecified 
- Misspecified 

The "auto" feature is designing an automathic dispatching mechanism when vcov is called 
`vcov(ve::VarianceEstimator; kwargs...)`


### 2.2 Manual assembly (matrix path)
Sometime it might be useful to manually assample an estimator. In this case, we simply provide 
```julia
vcov(ve::VarianceEstimator, Z::AbstractMatrix;
    form::Symbol,                       # required here
    jacobian::Union{Nothing,AbstractMatrix}=nothing,        # G (m×k)
    objective_hessian::Union{Nothing,AbstractMatrix}=nothing, # H (k×k)
    W::Union{Nothing,AbstractMatrix}=nothing,               # GMM weight (m×m)
    rcond_tol::Real = 1e-12) -> Matrix{Float64}
```

As above, a better alternative is 
```
vcov(ve::VarianceEstimator, form::VarianceEstimatorType, Z::AbstractMatrix;
    jacobian::Union{Nothing,AbstractMatrix}=nothing,        # G (m×k)
    objective_hessian::Union{Nothing,AbstractMatrix}=nothing, # H (k×k)
    W::Union{Nothing,AbstractMatrix}=nothing,               # GMM weight (m×m)
    rcond_tol::Real = 1e-12) -> Matrix{Float64}
```

the dispatch then takes care of which is possible when one passes only the jacobain for instance (the sandwich is not possible in this case) and the form binds only if everything has been passed. 


## 3) Model Integration Hooks (duck-typed)

Third-party estimator objects should (ideally) implement:

```julia
CovarianceMatrices.momentmatrix(model)                         # n × m (at θ̂_0)
CovarianceMatrices.momentmatrix(model, θ::AbstractVector)      # n × m 
CovarianceMatrices.jacobian(model)                             # m × k
CovarianceMatrices.objective_hessian(model)                    # k × k
CovarianceMatrices.weight_matrix(model)                        # optional: W (m × m) (only for inefficient GMM)
StatsBase.coef(model)                                          # θ̂
```
## 4) Exact Formulas (by `form`)

### 4.1 M-like (exactly identified, m = k)

* `form = :information` (correctly specified, iid MLE)

  $$
  V \;=\; H^{-1}.
  $$

  Requires `objective_hessian(model)` (or uses `G` if provided and symmetric).

* `form = :robust` (misspecified / HC / HAC / CR / DK)

  $$
  V \;=\; G^{-1}\,\Omega\,G^{-T}.
  $$

> Identity: with m=k, $(G'\Omega^{-1}G)^{-1} \equiv G^{-1}\Omega G^{-T}$.

### 4.2 GMM-like (overidentified, m > k)

* `form = :correctly_specified` (optimal weight, correct moments)

  $$
  V \;=\; (G'\,\Omega^{-1}\,G)^{-1}.
  $$

  (We compute it without forming $\Omega^{-1}$ explicitly; see §7.)

* `form = :misspecified` (robust to misspecified moments and dependence)

  $$
  V \;=\; (G' W G)^{-1} \, (G' W\,\Omega\,W G) \, (G' W G)^{-1}.
  $$

  * If `W === nothing`, default to $W = \Omega^{-1}$ (then it collapses to `:correctly_specified`).
  * If the model provides `weight_matrix(model)`, we use that.

## 5) When Methods Are Missing

When "some" method are missing you should determine which form of the variance can be estimated and which cannot. 


## Numerically Stable Implementations (no `inv`)

Prefer a numerically scale optimization.  We implement each form using **factorizations/solves**, not matrix inverses. Outline:

### 7.1 Helper: SPD-ish solver / pseudo-inverse

* Try `cholesky(Symmetric(A); check=true)` → triangular solves.
* Fallback to SVD-based pseudo-inverse when needed (`rcond_tol` cutoff).
* Always symmetrize final covariances: `Symmetric(V)`.

### 7.2 `:correctly_specified` (GMM)

Compute $V=(G'\Omega^{-1}G)^{-1}$ without forming $\Omega^{-1}$:

1. Factor $\Omega \approx C C'$ via Cholesky if SPD; else use an SVD-backed solver `C \ ·`.
2. Solve $C X = G$ → $X = C^{-1} G$.
3. $K = X' X = G' \Omega^{-1} G$.
4. Factor $K$ (Cholesky or SVD) and solve $K^{-1}$ via triangular/SVD solves.

### 7.3 `:misspecified` (general robust GMM)

Let $K = G' W G$, $B = G' W \Omega W G$.

* If `W` not given, **compute `W := Ω^{-1}`** via SPD/SVD solver (no explicit inverse).
* Compute `K` and `B`; solve $V = K^{-1} B K^{-1}$ using Cholesky (or SVD (pinv) fallback) with **two solves**.

### 7.4 `:robust` (M-like)

Solve $V = G^{-1} \Omega G^{-T}$ with **one factorization** of $G$ (square):

* Use `qr(G)` (robust for non-SPD).
  Solve $G X = \Omega$ (left divide), then $X = V G'$ → $V = X / G'$.

### 7.5 `:information` (M-like)

Solve $V = H^{-1}$:

* Prefer `ldlt`/`cholesky(Symmetric(H); check=false)`; SVD (pinv) fallback if needed.

## Checks & Warnings

* **Shape checks** (`check=true`):

  * `Z: n×m`, `G: m×k`, `H: k×k`.
  * M-like requires `m == k`.
  * `:information` requires either `H` or `G` (if symmetric).

* **Warnings** (`warn=true`):

  * Using HAC/CR/DK kernels with `:information` → warn (user likely meant `:robust`).
  * `:misspecified` with `W === nothing` → note it reduces to `:correctly_specified`.
  * non invertibility or non psd and use of pinv in the calculation of the variance. 


## 9) Examples

### 9.1 M-like (Probit/Logit) — correctly specified

```julia
vc = vcov(IID(), Information(), probit)  # V = H^{-1}
```

### 9.2 M-like — robust HAC

```julia
vc = vcov(Bartlett(5), Robust(), probit) # V = G^{-1} Ω G^{-T}
```

### 9.3 GMM — correctly specified (preferred default)

```julia
vc = vcov(HC1(), CorrectlySpecified(), mygmm) # (G' Ω^{-1} G)^{-1}
```

### 9.4 GMM — misspecified (full robust GMM with user W)

```julia
W = weight_matrix(mygmm)             # e.g., from first-step
vc = vcov(Bartlett(5), Misspecified(), mygmm, W=W)
```

### 9.5 Manual matrices

```julia
Z = momentmatrix(mygmm)
G = jacobian(mygmm)
vcov(Bartlett(5), Z; jacobian=G) ## Understand what to do depending on size. 

but in this case just simple to do 
B = aVar(Bartlett(5), Z)
And the if the user has J^{-1} can form
J^{-1}BJ^{-1}

## Scaling issue

Be aware that aVar return an estimate of the variance  sqrt{n}(mean(Z)). Thus the jacobian and the hessian thay have all to be scaled by n as well. 


## 11) Backwards Compatibility & Migration

* Keep `aVar(ve, Z)` unchanged.
* Deprecate any legacy `bread(model)` usage:

  * For M-like, instruct to implement `jacobian(model)` (equals the old bread).
  * For GMM-like, provide `objective_hessian(model)` only if they really need it; most usage won’t require `H`.
* Existing `GLM` extension can implement `momentmatrix/jacobian/objective_hessian` internally, preserving current behavior.

## 11) Testing & Benchmarks

* **Equivalences**:

  * For m=k: `:robust` equals `:correctly_specified` numerically (identity above).
  * For GMM: `:misspecified` with `W=Ω^{-1}` equals `:correctly_specified`.
* **Stability**:

  * Semi-definite Ω (few clusters, small T): SVD fallback, finite results.
  * Nearly singular $G'WG$: SVD fallback.
* **Performance**:

  * No `inv` in hot paths.
  * Reuse factorizations where possible; match/beat current `aVar` timings.



## 12) Developer Guidance (1-page cheatsheet)

To integrate your estimator `MyModel`:

1. Implement:

   ```julia
   StatsBase.coef(model)::AbstractVector
   CovarianceMatrices.momentmatrix(model)::AbstractMatrix         # n×m
   CovarianceMatrices.jacobian(model)::AbstractMatrix             # m×k
   ```

   Optional:

   ```julia
   CovarianceMatrices.momentmatrix(model, θ)::AbstractMatrix
   CovarianceMatrices.objective_hessian(model)::AbstractMatrix    # k×k
   CovarianceMatrices.weight_matrix(model)::AbstractMatrix        # m×m
   ```

2. Get covariance:

   * M-like: `vcov(Bartlett(5), model; form=:robust)` or `vcov(IID(), model; form=:information)`
   * GMM-like: `vcov(HC1(), model; form=:correctly_specified)` or `vcov(Bartlett(5), model; form=:misspecified, W=W)`

3. If you only have matrices, use `vcov(., ::AbstractMatrix,...)`.

