# HAC Estimators

HAC (Heteroskedasticity and Autocorrelation Consistent) estimators provide robust covariance matrices for time series data in the presence of both conditional heteroskedasticity and serial correlation.

## Mathematical Foundation

For time series data the long-run covariance matrix is

```math
\Omega = \sum_{j=-\infty}^{\infty} \mathbb{E}[g(z_t, \theta_0) g(z_{t-j}, \theta_0)'].
```

HAC estimators approximate it with a weighted sum of sample autocovariances:

```math
\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^{T-1} k\!\left(\frac{j}{S_T}\right) \left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right],
```

where $\hat{\Gamma}_j = \frac{1}{T} \sum_{t=j+1}^T g_t g_{t-j}'$ are the sample autocovariances, $k(\cdot)$ is the kernel, and $S_T$ is the bandwidth. The kernel downweights distant autocovariances; the bandwidth controls how fast the weights decay.

## Available Kernels

```@docs
Bartlett
CovarianceMatrices.BartlettKernel
Parzen
CovarianceMatrices.ParzenKernel
QuadraticSpectral
CovarianceMatrices.QuadraticSpectralKernel
Truncated
CovarianceMatrices.TruncatedKernel
TukeyHanning
CovarianceMatrices.TukeyHanningKernel
```

### Bartlett (Triangular)

```math
k(x) = \begin{cases} 1 - |x| & |x| \leq 1 \\ 0 & \text{otherwise} \end{cases}
```

The triangular weights guarantee a positive semi-definite estimate (Newey and West, 1987). This is the default in most applied work.

### Parzen

```math
k(x) = \begin{cases}
1 - 6x^2 + 6|x|^3 & |x| \leq 1/2 \\
2(1-|x|)^3 & 1/2 < |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

A smoother, higher-order kernel that is also positive semi-definite.

### Quadratic Spectral

```math
k(x) = \frac{25}{12\pi^2 x^2}\left[\frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\right]
```

Unbounded support; Andrews (1991) shows it minimizes the asymptotic mean squared error among kernels in its class.

### Truncated (Uniform)

```math
k(x) = \begin{cases} 1 & |x| \leq 1 \\ 0 & \text{otherwise} \end{cases}
```

Equal weights up to the bandwidth. It does not guarantee a positive semi-definite estimate.

### Tukey-Hanning

```math
k(x) = \begin{cases} \frac{1 + \cos(\pi x)}{2} & |x| \leq 1 \\ 0 & \text{otherwise} \end{cases}
```

## Bandwidth Selection

The bandwidth $S_T$ is the kernel's type parameter. The package supplies two data-driven rules and a fixed option.

```@docs
Andrews
NeweyWest
```

### Andrews

The Andrews (1991) rule chooses the bandwidth that minimizes the asymptotic mean squared error,

```math
S_T^* = \alpha_q \left(\frac{\hat{\sigma}^2_1}{\hat{\sigma}^2_0}\right)^{2/(2q+1)} T^{1/(2q+1)},
```

where $\alpha_q$ depends on the kernel and the spectral quantities $\hat{\sigma}^2_0, \hat{\sigma}^2_1$ are estimated from an AR(1) approximation to each moment series.

### Newey-West

The Newey-West (1994) rule selects the bandwidth from a preliminary count of autocovariances rather than an AR(1) fit. It is cheaper than the Andrews rule and avoids the AR(1) approximation.

## Usage

The estimator is the kernel together with its bandwidth rule, written `Kernel{Rule}()`. It attaches to a fitted model through `stderror` and `vcov`. The monthly US `economics` series regresses the personal saving rate on the unemployment level, where the residuals are strongly serially correlated.

```@example hac
using CovarianceMatrices, GLM, RDatasets

econ = RDatasets.dataset("ggplot2", "economics")
model = lm(@formula(PSavert ~ Unemploy), econ)
stderror(model)            # classical standard errors, no serial-correlation correction
```

```@example hac
stderror(Bartlett{Andrews}(), model)
```

The robust standard errors are several times the classical ones, reflecting the persistence the classical formula ignores. The choice of kernel matters much less than the decision to correct at all:

```@example hac
using DataFrames
DataFrame(
    coef = coefnames(model),
    classical = stderror(model),
    bartlett = stderror(Bartlett{Andrews}(), model),
    parzen = stderror(Parzen{Andrews}(), model),
    qs = stderror(QuadraticSpectral{Andrews}(), model),
)
```

A fixed bandwidth is set by passing an integer to the kernel constructor:

```@example hac
stderror(Bartlett(6), model)
```

### Prewhitening

Prewhitening fits a VAR(1) to the moments, applies the kernel to the residuals, and recolors the result. It reduces bias when the data are highly persistent (Andrews and Monahan, 1992):

```@example hac
stderror(Bartlett{Andrews}(), model; prewhite=true)
```

### Inspecting the bandwidth

`optimalbw` returns the bandwidth a rule selects for a given moment matrix, without forming the covariance:

```@example hac
g = momentmatrix(model)
optimalbw(Bartlett{Andrews}(), g)
```

```@example hac
optimalbw(Bartlett{NeweyWest}(), g)
```

## References

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". *Econometrica*, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". *Econometrica*, 59(3), 817-858.
- Andrews, D.W.K. and Monahan, J.C. (1992). "An Improved Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimator". *Econometrica*, 60(4), 953-966.
- Newey, W.K. and West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation". *Review of Economic Studies*, 61(4), 631-653.
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". *Handbook of Statistics*, 15, 291-341.
