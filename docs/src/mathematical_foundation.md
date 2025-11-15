# Mathematical Foundation

This section provides the mathematical foundation underlying the covariance matrix estimators implemented in CovarianceMatrices.jl.

## General Framework

### The Estimation Problem

Consider a statistical model with parameter vector $\theta \in \Theta \subset \mathbb{R}^k$ estimated from data $\{z_t\}_{t=1}^T$. The estimator $\hat{\theta}$ satisfies moment conditions:

```math
\frac{1}{T} \sum_{t=1}^T g(z_t, \hat{\theta}) = 0
```

where $g: \mathcal{Z} \times \Theta \rightarrow \mathbb{R}^m$ are the moment conditions.

### Asymptotic Distribution

Under regularity conditions, the estimator has the asymptotic distribution:

```math
\sqrt{T}(\hat{\theta} - \theta_0) \stackrel{d}{\rightarrow} \mathcal{N}(0, V)
```

The variance $V$ depends on the specification and the form of dependence in the data:

#### MLikeModel: Maximum Likelihood-type Models

For correctly specified models with $m = k$ moment conditions (exactly identified):

```math
V = G^{-1} \Omega G^{-1}
```

This applies to:

- **Maximum Likelihood Estimation (MLE)**: Score-based inference
- **Method of Moments**: Exactly identified systems
- **Pseudo-MLE**: Quasi-likelihood approaches

**Properties**:

- Achieves Cramér-Rao lower bound under correct specification
- Requires $m = k$ (number of moments equals number of parameters)
- $G$ is typically the expected Hessian or Fisher information matrix

**Usage in CovarianceMatrices.jl**:
For MLE-type problems, the covariance estimator computes the "bread-meat-bread" sandwich with symmetric bread matrices.

#### GMMLikeModel: Generalized Method of Moments Models

For overidentified or robust inference with $m \geq k$ moment conditions:

```math
V = (G'WG)^{-1} G' W \Omega W G (G'WG)^{-1}
```

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

**Usage in CovarianceMatrices.jl**:
Most estimators in the package (HC, HAC, CR) implement this form with $W = I$, providing robust inference without efficiency claims.

where:

- $G = \mathbb{E}[\nabla_\theta g(z_t, \theta_0)]$ is the gradient matrix (score)
- $\Omega = \text{Var}[\frac{1}{\sqrt{T}} \sum_{t=1}^T g(z_t, \theta_0)]$ is the long-run covariance matrix
- $W$ is an optional weight matrix (identity for standard sandwich)

## Types of Dependence and Estimators

The key challenge lies in estimating $\Omega$, which depends on the dependence structure in the data.

### 1. Independent and Identically Distributed Data

For i.i.d. data: $\Omega = \mathbb{E}[g(z_t, \theta_0) g(z_t, \theta_0)']$

**Estimator**: Simple sample covariance

```math
\hat{\Omega} = \frac{1}{T} \sum_{t=1}^T g_t g_t'
```

### 2. Heteroskedasticity Only

For data with conditional heteroskedasticity but no serial correlation:

```math
\Omega = \mathbb{E}[g(z_t, \theta_0) g(z_t, \theta_0)' | \mathcal{F}_{t-1}]
```

**Estimators**: HC/HR family

```math
\hat{\Omega}_{HC_j} = \frac{1}{T} \sum_{t=1}^T \phi_j(h_t) g_t g_t'
```

where $\phi_j(h_t)$ are leverage-dependent adjustment factors:

- $HC_0$: $\phi_0(h_t) = 1$
- $HC_1$: $\phi_1(h_t) = \frac{T}{T-k}$
- $HC_2$: $\phi_2(h_t) = \frac{1}{1-h_t}$
- $HC_3$: $\phi_3(h_t) = \frac{1}{(1-h_t)^2}$

### 3. Serial Correlation

For serially correlated data:

```math
\Omega = \sum_{j=-\infty}^{\infty} \mathbb{E}[g(z_t, \theta_0) g(z_{t-j}, \theta_0)']
```

#### HAC Estimators

**General Form**:

```math
\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^{m} k\left(\frac{j}{S_T}\right) \left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right]
```

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

#### VARHAC Estimator

**Theoretical Foundation**:

VARHAC (Vector Autoregression HAC) provides a parametric alternative to kernel-based HAC estimation. Instead of selecting kernels and bandwidths, VARHAC models the serial correlation structure directly through a Vector Autoregression.

**The VAR Representation**:

The moment conditions are modeled as a VAR(p) process:

```math
g_t = c + \sum_{j=1}^p A_j g_{t-j} + \varepsilon_t
```

where:

- $g_t \in \mathbb{R}^m$ are the moment conditions at time $t$
- $c \in \mathbb{R}^m$ is a vector of constants
- $A_j \in \mathbb{R}^{m \times m}$ are autoregressive coefficient matrices
- $\varepsilon_t \in \mathbb{R}^m$ are i.i.d. innovations with $\mathbb{E}[\varepsilon_t] = 0$ and $\text{Var}[\varepsilon_t] = \Sigma_\varepsilon$
- $p$ is the lag order, selected via information criteria (AIC/BIC)

**Long-run Covariance Computation**:

From the Wold representation theorem, the long-run covariance matrix (spectral density at frequency zero) is:

```math
\hat{\Omega}_{VARHAC} = \left(I - \sum_{j=1}^p \hat{A}_j\right)^{-1} \hat{\Sigma}_\varepsilon \left(I - \sum_{j=1}^p \hat{A}_j'\right)^{-1}
```

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

**Theoretical Properties**:

- **Consistency**: $\hat{\Omega}_{VARHAC} \stackrel{p}{\rightarrow} \Omega$ under regularity conditions
- **Asymptotic equivalence**: Equivalent to optimal HAC under correct VAR specification
- **Robustness**: Provides consistent estimates even under VAR misspecification
- **Finite sample performance**: Often superior to HAC in small to moderate samples

**Practical Implementation**:
The package allows flexible lag selection strategies:

- **SameLags**: Same maximum lag for all variables
- **FixedLags**: User-specified fixed lag order
- **AutoLags**: Automatic lag selection based on sample size
- **DifferentOwnLags**: Variable-specific lag orders

#### Smoothed Moments Estimator

Smith's method smooths moment conditions before taking outer products:

```math
\tilde{g}_t = \sum_{j=-m_S}^{m_S} w_j g_{t-j}
```

```math
\hat{\Omega}_{SM} = c \cdot \frac{1}{T} \sum_{t=1}^T \tilde{g}_t \tilde{g}_t'
```

where:

- $w_j = \frac{1}{S_T} k\left(\frac{j}{S_T}\right)$ are smoothing weights
- $m_S = \lfloor S_T \rfloor$ is the smoothing truncation lag
- $c$ is a normalization constant

**Relationship to HAC Estimators**:
Smoothed moments are asymptotically equivalent to corresponding HAC estimators:

- **Uniform Kernel** (Smoothed) ≡ **Truncated Kernel** (HAC)
- **Triangular Kernel** (Smoothed) ≡ **Bartlett Kernel** (HAC)

**Bandwidth Interpretation**:
Both methods use the same bandwidth parameter $m_T$ with identical interpretations:

- Controls the maximum lag included: $m = \lfloor m_T \rfloor$
- Determines kernel argument scaling: $k(2j/(2m_T+1)$
- Same optimal bandwidth rates apply to both approaches

The key difference is computational: smoothed moments first smooths moment conditions then takes outer products, while HAC first computes autocovariances then applies kernel weighting. Both approaches yield identical asymptotic results when using the same bandwidth.

**Advantages**:

- Automatic positive semi-definiteness
- Efficient kernel-based computation
- Optimal bandwidth scaling: $S_T = c \cdot T^{\alpha}$ where $\alpha = 1/3$ (Uniform) or $1/5$ (Triangular)

### 4. Cluster Correlation

For data clustered by groups $g \in \{1, \ldots, G\}$:

```math
\Omega = \sum_{g=1}^G \mathbb{E}\left[\left(\sum_{t \in g} g(z_t, \theta_0)\right)\left(\sum_{t \in g} g(z_t, \theta_0)\right)'\right]
```

**CR Estimators**:

```math
\hat{\Omega}_{CR_j} = \frac{G}{G-1} \sum_{g=1}^G \phi_j(g) \hat{u}_g \hat{u}_g'
```

where $\hat{u}_g = \sum_{t \in g} g_t$ and $\phi_j(g)$ are small-sample corrections.

### 5. Panel Data with Spatial-Temporal Dependence

For panel data with both cross-sectional and time dependence:

```math
\Omega = \sum_{h=-H}^H \sum_{s=-S}^S k_1\left(\frac{h}{H_T}\right) k_2\left(\frac{s}{S_T}\right) \Gamma_{h,s}
```

**Driscoll-Kraay Estimator**: Uses separate kernels for time and spatial dimensions.

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

## References

**General Theory**:

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". _Econometrica_, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". _Econometrica_, 59(3), 817-858.

**Specific Methods**:

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". _Econometrica_, 48(4), 817-838.
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". _Handbook of Statistics_, Vol. 15.
- Smith, R.J. (2005). "Automatic positive semidefinite HAC covariance matrix and GMM estimation". _Econometric Theory_, 21(1), 158-170.
