# HAC Estimators

HAC (Heteroskedasticity and Autocorrelation Consistent) estimators provide robust covariance matrices for time series data in the presence of both conditional heteroskedasticity and serial correlation.

## Mathematical Foundation

For time series data, the long-run covariance matrix is:

```math
\Omega = \sum_{j=-\infty}^{\infty} \mathbb{E}[g(z_t, \theta_0) g(z_{t-j}, \theta_0)']
```

HAC estimators approximate this using:

```math
\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^{q} k\left(\frac{j}{S_T}\right) \left[\hat{\Gamma}_j + \hat{\Gamma}_j'\right]
```

where:
- $\hat{\Gamma}_j = \frac{1}{T} \sum_{t=j+1}^T g_t g_{t-j}'$ are sample autocovariances
- $k(\cdot)$ is the kernel function
- $S_T$ is the bandwidth parameter
- $q$ is the truncation lag

## Available Kernels

### Bartlett (Triangular) Kernel

```@docs
Bartlett
```

The Bartlett kernel is defined as:
```math
k(x) = \begin{cases}
1 - |x| & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**
- Guarantees positive semi-definiteness
- Most commonly used in practice
- Good finite sample properties

**Usage:**
```julia
# Automatic bandwidth selection (Andrews)
bart_auto = Bartlett{Andrews}()

# Fixed bandwidth
bart_fixed = Bartlett(5)

# Newey-West bandwidth
bart_nw = Bartlett{NeweyWest}()
```

### Parzen Kernel

```@docs
Parzen
```

The Parzen kernel uses a more complex weighting scheme:
```math
k(x) = \begin{cases}
1 - 6x^2 + 6|x|^3 & \text{if } |x| \leq 1/2 \\
2(1-|x|)^3 & \text{if } 1/2 < |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**
- Guarantees positive semi-definiteness
- Higher order kernel (smoother)
- Better for data with strong serial correlation

**Usage:**
```julia
# Automatic bandwidth selection
parzen_auto = Parzen{Andrews}()

# Fixed bandwidth
parzen_fixed = Parzen(8)
```

### Quadratic Spectral Kernel

```@docs
QuadraticSpectral
```

The Quadratic Spectral kernel is unbounded:
```math
k(x) = \frac{25}{12\pi^2 x^2}\left[\frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\right]
```

**Properties:**
- Does not guarantee positive semi-definiteness in finite samples
- Theoretically optimal rate properties
- Can produce negative eigenvalues

**Usage:**
```julia
# Recommended: Use with Andrews bandwidth
qs_auto = QuadraticSpectral{Andrews}()

# Fixed bandwidth (use with caution)
qs_fixed = QuadraticSpectral(10)
```

### Truncated (Uniform) Kernel

```@docs
Truncated
```

The truncated kernel provides simple equal weighting:
```math
k(x) = \begin{cases}
1 & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**
- Guarantees positive semi-definiteness
- Simplest kernel
- Can be choppy for highly persistent data

### Tukey-Hanning Kernel

```@docs
TukeyHanning
```

The Tukey-Hanning kernel uses cosine weighting:
```math
k(x) = \begin{cases}
\frac{1 + \cos(\pi x)}{2} & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
```

**Properties:**
- Guarantees positive semi-definiteness
- Smooth weighting scheme
- Good for seasonal data

## Bandwidth Selection

The choice of bandwidth $S_T$ is crucial for HAC estimation. Two main approaches are available:

### Andrews Automatic Bandwidth

```@docs
Andrews
```

The Andrews method selects bandwidth to minimize asymptotic MSE:

```math
S_T^* = \alpha_2 \left(\frac{\hat{\sigma}^2_1}{\hat{\sigma}^2_0}\right)^{2/5} T^{2/5}
```

where $\alpha_2$ depends on the kernel and $\hat{\sigma}^2_0, \hat{\sigma}^2_1$ are estimated from a VAR(1) approximation.

**Advantages:**
- Theoretically justified
- Adapts to data characteristics
- Available for all kernels

**Usage:**
```julia
# Most common specification
hac_andrews = Bartlett{Andrews}()
Ω = aVar(hac_andrews, X)
```

### Newey-West Rule of Thumb

```@docs
NeweyWest
```

The Newey-West rule uses a two-step process:

1. **Preliminary VAR estimation** with lag truncation `l`:
   - Bartlett: `l = floor(4 * (T/100)^(2/9))`
   - Parzen: `l = floor(4 * (T/100)^(4/25))`

2. **Bandwidth formula**:
   - Bartlett: `S_T* = 1.1447 * (a1/a0)^(2/3) * T^(1/3)`
   - Parzen: `S_T* = 2.6614 * (a2/a0)^(2/5) * T^(1/5)`

where `a0, a1, a2` are autocorrelation-based parameters from the preliminary VAR(l).

**Advantages:**
- Data-adaptive through preliminary estimation
- Kernel-specific optimization
- Robust finite sample performance

**Limitations:**
- More complex than simple rule-of-thumb
- Requires preliminary VAR estimation step

**Usage:**
```julia
# Newey-West bandwidth
hac_nw = Bartlett{NeweyWest}()
Ω = aVar(hac_nw, X)
```

### Fixed Bandwidth

For specific applications, you may want to use a fixed bandwidth:

```julia
# Fixed bandwidth of 6
hac_fixed = Bartlett(6)
Ω = aVar(hac_fixed, X)

# Extract current bandwidth
current_bw = bandwidth(hac_fixed)
```

## Practical Guidelines

### Kernel Selection

1. **Default choice**: `Bartlett{Andrews}()`
   - Most widely used and tested
   - Guaranteed positive semi-definite
   - Good finite sample properties

2. **Strong serial correlation**: `Parzen{Andrews}()`
   - Higher order kernel
   - Better for persistent data

3. **Theoretical optimality**: `QuadraticSpectral{Andrews}()`
   - Use with caution due to PSD issues
   - Check eigenvalues of resulting matrix

4. **Simple applications**: `Truncated{Andrews}()`
   - Interpretable as simple averaging
   - Good for weakly correlated data

### Bandwidth Selection

1. **Adaptive approach**: Use `{Andrews}` for automatic selection
2. **Conservative approach**: Use `{NeweyWest}` for simpler rule
3. **Sensitivity analysis**: Try both and compare results
4. **Fixed bandwidth**: Use when you have domain knowledge

### Example: Comprehensive HAC Analysis

```julia
using CovarianceMatrices, Random
Random.seed!(123)

# Generate AR(1) time series
T = 200
ρ = 0.7
X = zeros(T, 2)
for t in 2:T
    X[t, :] = ρ * X[t-1, :] + randn(2)
end

# Compare different HAC estimators
estimators = [
    ("Bartlett-Andrews", Bartlett{Andrews}()),
    ("Bartlett-NeweyWest", Bartlett{NeweyWest}()),
    ("Bartlett-Fixed", Bartlett(6)),
    ("Parzen-Andrews", Parzen{Andrews}()),
    ("Quadratic Spectral", QuadraticSpectral{Andrews}()),
]

for (name, est) in estimators
    Ω = aVar(est, X)
    eigenvals = eigvals(Ω)
    println("$name:")
    println("  Trace: $(round(tr(Ω), digits=3))")
    println("  Min eigenvalue: $(round(minimum(eigenvals), digits=4))")
    println("  Condition number: $(round(cond(Ω), digits=1))")

    # Extract bandwidth if available
    if hasmethod(bandwidth, (typeof(est),))
        println("  Bandwidth: $(round(bandwidth(est)[1], digits=2))")
    end
    println()
end
```

### Prewhitening

HAC estimators can benefit from prewhitening, which fits a VAR(1) model to remove first-order serial correlation:

```julia
# Without prewhitening
Ω_no_prewhite = aVar(Bartlett{Andrews}(), X)

# With prewhitening
Ω_prewhite = aVar(Bartlett{Andrews}(), X; prewhite=true)

println("Without prewhitening: trace = $(round(tr(Ω_no_prewhite), digits=3))")
println("With prewhitening: trace = $(round(tr(Ω_prewhite), digits=3))")
```

### Diagnostic Tools

```julia
# Check bandwidth selection
hac_est = Bartlett{Andrews}()
_, _, bw = workingoptimalbw(hac_est, X)
println("Selected bandwidth: $(round(bw, digits=2))")

# Compare with rule of thumb
bw_nw = 4 * (T/100)^(2/9)
println("Newey-West bandwidth: $(round(bw_nw, digits=2))")

# Bandwidth sensitivity analysis
bandwidths = [3, 5, 8, 12]
for bw in bandwidths
    Ω = aVar(Bartlett(bw), X)
    println("Bandwidth $bw: trace = $(round(tr(Ω), digits=3))")
end
```

## Performance Considerations

HAC estimators vary significantly in computational cost:

1. **Fastest**: Fixed bandwidth estimators
2. **Medium**: NeweyWest automatic selection
3. **Slower**: Andrews automatic selection (requires VAR estimation)
4. **Alternative**: Consider VARHAC for automatic and fast computation

For large datasets or frequent estimation, consider using VARHAC as an alternative that avoids bandwidth selection entirely.

## References

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". *Econometrica*, 55(3), 703-708.
- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". *Econometrica*, 59(3), 817-858.
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust Covariance Matrix Estimation". *Handbook of Statistics*, 15, 291-341.