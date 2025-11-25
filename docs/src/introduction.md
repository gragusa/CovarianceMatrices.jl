# Introduction and Mathematical Foundation

This document provides the mathematical foundations underlying robust covariance matrix estimation, as implemented in [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl/).

## The Estimation Problem

Consider a sequence of $k$-dimensional random vectors $\{X_t, 1 \leq t \leq T\}$. Define the sample mean:

$$
\bar{X}_T = \frac{1}{T}\sum_{t=1}^T X_t
$$

Let $\mu_T = E(\bar{X}_T)$ denote the expected value of the sample mean. Under appropriate regularity conditions, the standardized sample mean satisfies:

$$
\sqrt{T}\Sigma_T^{-1/2}(\bar{X}_T - \mu_T) \xrightarrow{d} N(0, I_k)
$$

where $\Sigma_T$ is the asymptotic variance-covariance matrix and $I_k$ is the $k$-dimensional identity matrix. The fundamental estimation problem is: **how do we estimate $\Sigma_T$?**

The answer depends critically on the correlation structure of the sequence $\{X_t\}$.

## General Framework for Statistical Models

### Parameter Estimation

Consider a statistical model with parameter vector $\theta \in \Theta \subset \mathbb{R}^k$ estimated from data $\{z_t\}_{t=1}^T$. The estimator $\hat{\theta}$ typically satisfies moment conditions:

$$
\frac{1}{T} \sum_{t=1}^T g(z_t, \hat{\theta}) = 0
$$

where $g: \mathcal{Z} \times \Theta \rightarrow \mathbb{R}^m$ are the moment conditions.

### Asymptotic Distribution

Under regularity conditions, the estimator has the asymptotic distribution:

$$
\sqrt{T}(\hat{\theta} - \theta_0) \stackrel{d}{\rightarrow} \mathcal{N}(0, V)
$$

The variance $V$ depends on the model specification and the form of dependence in the data:

#### MLikeModel: Maximum Likelihood-type Models

For correctly specified models with $m = k$ moment conditions (exactly identified):

$$
V = G^{-1} \Omega G^{-1}
$$

This applies to:
- **Maximum Likelihood Estimation (MLE)**: Score-based inference
- **Method of Moments**: Exactly identified systems
- **Pseudo-MLE**: Quasi-likelihood approaches

**Properties**:
- Achieves Cramér-Rao lower bound under correct specification
- Requires $m = k$ (number of moments equals number of parameters)
- $G$ is typically the expected Hessian or Fisher information matrix

#### GMMLikeModel: Generalized Method of Moments Models

For overidentified or robust inference with $m \geq k$ moment conditions:

$$
V = (G'WG)^{-1} G' W \Omega W G (G'WG)^{-1}
$$

This applies to:
- **Generalized Method of Moments (GMM)**: Overidentified systems
- **Robust Regression**: Heteroskedasticity-robust inference
- **Two-Step Estimators**: Instrumental variables, 2SLS
- **Misspecified Models**: Quasi-maximum likelihood

**Properties**:
- Allows $m > k$ (overidentification)
- Robust to model misspecification
- Optimal choice: $W = \Omega^{-1}$ (efficient GMM)
- Common choice: $W = I$ (standard sandwich estimator)

where:
- $G = \mathbb{E}[\nabla_\theta g(z_t, \theta_0)]$ is the gradient matrix (score)
- $\Omega = \text{Var}[\frac{1}{\sqrt{T}} \sum_{t=1}^T g(z_t, \theta_0)]$ is the long-run covariance matrix
- $W$ is an optional weight matrix (identity for standard sandwich)

## Types of Dependence and Estimation Methods

The key challenge lies in estimating $\Omega$, which depends on the dependence structure in the data.

### 1. Independent and Identically Distributed Data

When $\{X_t\}$ is an i.i.d. sequence, the variance $\Sigma_T$ has a simple form:

$$
\Omega = \mathbb{E}[g(z_t, \theta_0) g(z_t, \theta_0)']
$$

**Estimator**: Simple sample covariance

$$
\hat{\Omega} = \frac{1}{T} \sum_{t=1}^T g_t g_t'
$$

This is the classical sample covariance matrix, and estimation is straightforward. In CovarianceMatrices.jl:

```julia
using CovarianceMatrices

# X is T × k matrix of observations
Σ_hat = aVar(Uncorrelated(), X; demean=true)
```

### 2. Heteroskedasticity Only

For cross-sectional data with conditional heteroskedasticity but no serial correlation:

$$
\Omega = \mathbb{E}[g(z_t, \theta_0) g(z_t, \theta_0)' | \mathcal{F}_{t-1}]
$$

**Estimators**: HC/HR family

$$
\hat{\Omega}_{HC_j} = \frac{1}{T} \sum_{t=1}^T \phi_j(h_t) g_t g_t'
$$

where $\phi_j(h_t)$ are leverage-dependent adjustment factors:

- $HC_0$: $\phi_0(h_t) = 1$
- $HC_1$: $\phi_1(h_t) = \frac{T}{T-k}$
- $HC_2$: $\phi_2(h_t) = \frac{1}{1-h_t}$
- $HC_3$: $\phi_3(h_t) = \frac{1}{(1-h_t)^2}$

**Example:**
```julia
# HC3 is generally recommended
Σ_hc3 = aVar(HC3(), X)

# Other variants: HC0, HC1, HC2, HC4, HC5
Σ_hc0 = aVar(HC0(), X)
```

**Reference:**
- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". *Econometrica*, 48(4), 817-838.

### 3. Serial Correlation (Time Series)

When the sequence $\{X_t\}$ exhibits serial correlation, estimating $\Sigma_T$ requires accounting for the **long-run covariance**:

$$
\Omega = \sum_{j=-\infty}^{\infty} \mathbb{E}[g(z_t, \theta_0) g(z_{t-j}, \theta_0)']
$$

This is the **spectral density at frequency zero**, capturing both contemporaneous variance and all autocovariances.

#### HAC Estimators

**General Form**:

$$
\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^{m} k\left(\frac{j}{S_T}\right) \left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right]
$$

where:
- $k(\cdot)$ is a kernel function
- $S_T$ is the bandwidth parameter
- $m = \lfloor S_T \rfloor$ is the truncation lag (maximum lag used)
- $\hat{\Gamma}_j = \frac{1}{T} \sum_{t=j+1}^T g_t g_{t-j}'$ are sample autocovariances

**Bandwidth Interpretation**:
For truncated and Bartlett/Parzen kernels, the bandwidth $S_T$ determines the number of lags included. Specifically:
- Total number of autocovariances used: $2m + 1$ where $m = \lfloor S_T \rfloor$
- These correspond to lags $j \in \{-m, -m+1, \ldots, -1, 0, 1, \ldots, m-1, m\}$
- The bandwidth $S_T$ acts as the "window width" parameter

**Kernel Functions**:

- **Bartlett**: $k(x) = 1 - |x|$ for $|x| \leq 1$
- **Parzen**: $k(x) = \begin{cases}
1 - 6x^2 + 6|x|^3 & \text{if } |x| \leq 1/2 \\
2(1-|x|)^3 & \text{if } 1/2 < |x| \leq 1
\end{cases}$
- **Quadratic Spectral**: $k(x) = \frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)$

**Bandwidth Selection**:

**Andrews Method**: Uses kernel-specific optimal rates:

- **Bartlett**: $S_T^* = 1.1447 \cdot (\hat{a}_1 \cdot T)^{1/3}$
- **Parzen**: $S_T^* = 2.6614 \cdot (\hat{a}_2 \cdot T)^{1/5}$
- **Quadratic Spectral**: $S_T^* = 1.3221 \cdot (\hat{a}_2 \cdot T)^{1/5}$
- **Truncated**: $S_T^* = 0.6611 \cdot (\hat{a}_2 \cdot T)^{1/5}$

**Newey-West Method**: Uses autocorrelation-based selection with kernel-dependent growth rates:

- **Bartlett**: $S_T^* = 1.1447 \cdot \left(\frac{\hat{a}_1}{\hat{a}_0}\right)^{2/3} \cdot T^{1/3}$
- **Other kernels**: $S_T^* = c_k \cdot \left(\frac{\hat{a}_2}{\hat{a}_0}\right)^{2/5} \cdot T^{1/5}$ where $c_k$ is kernel-specific

where $\hat{a}_j = 2\sum_{i=1}^l i^j \hat{\rho}_i$ are estimated autocorrelation-based parameters from a preliminary VAR(1) fit.

**Example:**
```julia
# Newey-West estimator (Bartlett kernel with automatic bandwidth)
Σ_nw = aVar(Bartlett{NeweyWest}(), X)

# Andrews optimal bandwidth
Σ_andrews = aVar(Parzen{Andrews}(), X)

# Fixed bandwidth
Σ_fixed = aVar(Bartlett(5), X)
```

**References:**
- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". *Econometrica*, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". *Econometrica*, 59(3), 817-858.

#### VARHAC Estimator

**Theoretical Foundation**:

VARHAC (Vector Autoregression HAC) provides a parametric alternative to kernel-based HAC estimation. Instead of selecting kernels and bandwidths, VARHAC models the serial correlation structure directly through a Vector Autoregression.

**The VAR Representation**:

The moment conditions are modeled as a VAR(p) process:

$$
g_t = c + \sum_{j=1}^p A_j g_{t-j} + \varepsilon_t
$$

where:
- $g_t \in \mathbb{R}^m$ are the moment conditions at time $t$
- $c \in \mathbb{R}^m$ is a vector of constants
- $A_j \in \mathbb{R}^{m \times m}$ are autoregressive coefficient matrices
- $\varepsilon_t \in \mathbb{R}^m$ are i.i.d. innovations with $\mathbb{E}[\varepsilon_t] = 0$ and $\text{Var}[\varepsilon_t] = \Sigma_\varepsilon$
- $p$ is the lag order, selected via information criteria (AIC/BIC)

**Long-run Covariance Computation**:

From the Wold representation theorem, the long-run covariance matrix (spectral density at frequency zero) is:

$$
\hat{\Omega}_{VARHAC} = \left(I - \sum_{j=1}^p \hat{A}_j\right)^{-1} \hat{\Sigma}_\varepsilon \left(I - \sum_{j=1}^p \hat{A}_j'\right)^{-1}
$$

This formula represents the limiting variance of $\frac{1}{\sqrt{T}} \sum_{t=1}^T g_t$ under the fitted VAR model.

**Lag Selection**:

The lag order $p$ is selected using information criteria:

- **AIC**: $\text{AIC}(p) = \log|\hat{\Sigma}_\varepsilon(p)| + \frac{2pm^2}{T}$
- **BIC**: $\text{BIC}(p) = \log|\hat{\Sigma}_\varepsilon(p)| + \frac{\log(T) \cdot pm^2}{T}$

where $\hat{\Sigma}_\varepsilon(p)$ is the residual covariance matrix from the VAR(p) fit.

**Key Advantages**:

1. **No bandwidth selection**: Eliminates subjective kernel and bandwidth choices
2. **Automatic positive semi-definiteness**: The formula guarantees $\hat{\Omega}_{VARHAC} \succeq 0$
3. **Data-adaptive**: Captures complex serial correlation patterns
4. **Parsimony**: BIC tends to select parsimonious models
5. **Computational efficiency**: No kernel computations required

**Example:**
```julia
# Automatic lag selection with BIC
Σ_varhac = aVar(VARHAC(:bic), X)

# Check selected lag order
varhac_est = VARHAC()
aVar(varhac_est, X)
println("Selected lags: ", order(varhac_est))
```

**Reference:**
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". *Handbook of Statistics*, Vol. 15.

#### Smoothed Moments Estimator

Smith's method smooths moment conditions before taking outer products:

$$
\tilde{g}_t = \sum_{j=-m_S}^{m_S} w_j g_{t-j}
$$

$$
\hat{\Omega}_{SM} = c \cdot \frac{1}{T} \sum_{t=1}^T \tilde{g}_t \tilde{g}_t'
$$

where:
- $w_j = \frac{1}{S_T} k\left(\frac{j}{S_T}\right)$ are smoothing weights
- $m_S = \lfloor S_T \rfloor$ is the smoothing truncation lag
- $c$ is a normalization constant

**Relationship to HAC Estimators**:
Smoothed moments are asymptotically equivalent to corresponding HAC estimators:

- **Uniform Kernel** (Smoothed) ≡ **Truncated Kernel** (HAC)
- **Triangular Kernel** (Smoothed) ≡ **Bartlett Kernel** (HAC)

**Advantages**:

- Automatic positive semi-definiteness
- Efficient kernel-based computation
- Optimal bandwidth scaling: $S_T = c \cdot T^{\alpha}$ where $\alpha = 1/3$ (Uniform) or $1/5$ (Triangular)

**Example:**
```julia
# Uniform smoothing (≡ Bartlett HAC asymptotically)
Σ_smooth_uniform = aVar(SmoothedMoments(UniformSmoother()), X)

# Triangular smoothing (≡ Parzen HAC asymptotically)
Σ_smooth_triangular = aVar(SmoothedMoments(TriangularSmoother()), X)
```

**Reference:**
- Smith, R.J. (2005). "Automatic positive semidefinite HAC covariance matrix and GMM estimation". *Econometric Theory*, 21(1), 158-170.

#### Equal Weighted Cosine (EWC)

The **EWC estimator** uses a basis function expansion:

$$
\hat{\Omega}_{EWC} = \sum_{b=1}^B w_b \hat{S}_b
$$

where $\hat{S}_b$ are spectral estimates at different frequencies. This provides a non-parametric alternative particularly useful for financial time series.

**Example:**
```julia
# EWC with 10 basis functions
Σ_ewc = aVar(EWC(10), X)
```

**Reference:**
- Lazarus, E., Lewis, D.J., Stock, J.H., and Watson, M.W. (2018). "HAR Inference: Recommendations for Practice". *Journal of Business & Economic Statistics*, 36(4), 541-559.

### 4. Cluster Correlation

For data clustered by groups $g \in \{1, \ldots, G\}$:

$$
\Omega = \sum_{g=1}^G \mathbb{E}\left[\left(\sum_{t \in g} g(z_t, \theta_0)\right)\left(\sum_{t \in g} g(z_t, \theta_0)\right)'\right]
$$

**CR Estimators**:

$$
\hat{\Omega}_{CR_j} = \frac{G}{G-1} \sum_{g=1}^G \phi_j(g) \hat{u}_g \hat{u}_g'
$$

where $\hat{u}_g = \sum_{t \in g} g_t$ and $\phi_j(g)$ are small-sample corrections.

**Example:**
```julia
# cluster is a vector of cluster IDs
Σ_cr1 = aVar(CR1(cluster), X)

# Two-way clustering
Σ_cr1_twoway = aVar(CR1((cluster1, cluster2)), X)
```

### 5. Panel Data with Spatial-Temporal Dependence

For panel data with both cross-sectional and time dependence:

$$
\Omega = \sum_{h=-H}^H \sum_{s=-S}^S k_1\left(\frac{h}{H_T}\right) k_2\left(\frac{s}{S_T}\right) \Gamma_{h,s}
$$

**Driscoll-Kraay Estimator**: Uses separate kernels for time and spatial dimensions.

**Example:**
```julia
Σ_dk = aVar(DriscollKraay(Bartlett{Andrews}(), tis=time_id, iis=unit_id), X)
```

## Computational Considerations

### Numerical Stability

All estimators implement numerically stable computation:

1. **Symmetric matrices**: Enforced symmetry via `Symmetric()` wrapper
2. **Condition number monitoring**: Warnings for ill-conditioned matrices
3. **Regularization**: Optional ridge-type regularization for near-singular cases
4. **Robust inversion**: Uses `pinv()` with appropriate tolerance when needed

### Performance Optimizations

1. **Memory efficiency**: In-place operations where possible
2. **Threading**: Parallel computation for computationally intensive estimators
3. **BLAS optimization**: Leverages optimized linear algebra routines
4. **Kernel-based computation**: Direct kernel evaluation vs. weight precomputation

## Finite Sample Properties

### Bias Corrections

Many estimators include finite sample bias corrections:

- **HC corrections**: Leverage-based adjustments (HC2, HC3, etc.)
- **CR corrections**: Small cluster adjustments
- **HAC corrections**: Bartlett kernel provides bias reduction

### Coverage Properties

Under appropriate regularity conditions:

1. **Consistency**: $\hat{\Omega} \stackrel{p}{\rightarrow} \Omega$ as $T \rightarrow \infty$
2. **Rate optimality**: HAC estimators achieve optimal MSE rates
3. **Coverage accuracy**: Confidence intervals have correct asymptotic coverage

## Summary: Which Estimator to Use?

| Data Structure | Recommended Estimator | Alternative |
|----------------|----------------------|-------------|
| IID data | `Uncorrelated()` | — |
| Cross-section with heteroskedasticity | `HC3()` | `HC2()`, `HC1()` |
| Time series | `VARHAC()` | `Bartlett{Andrews}()` |
| Time series (guaranteed PSD) | `SmoothedMoments()` | `VARHAC()` |
| Clustered data | `CR1(cluster)` | `CR2(cluster)` |
| Panel data | `DriscollKraay()` | Two-way `CR1()` |
| Financial time series | `EWC()` | `VARHAC()` |

## Next Steps

- **[Estimators](estimators/hac.md)**: Detailed documentation for each estimator
- **[GLM Tutorial](tutorials/glm_tutorial.md)**: Using CovarianceMatrices.jl with regression models
- **[Matrix Tutorial](tutorials/matrix_tutorial.md)**: Direct covariance matrix estimation
- **[Interface Tutorial](tutorials/interface_tutorial.md)**: Extending to custom model types
- **[API Reference](api.md)**: Complete API documentation

## References

**General Theory**:

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". *Econometrica*, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". *Econometrica*, 59(3), 817-858.

**Specific Methods**:

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". *Econometrica*, 48(4), 817-838.
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". *Handbook of Statistics*, Vol. 15.
- Smith, R.J. (2005). "Automatic positive semidefinite HAC covariance matrix and GMM estimation". *Econometric Theory*, 21(1), 158-170.
- Lazarus, E., Lewis, D.J., Stock, J.H., and Watson, M.W. (2018). "HAR Inference: Recommendations for Practice". *Journal of Business & Economic Statistics*, 36(4), 541-559.
