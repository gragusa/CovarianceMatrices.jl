# Driscoll-Kraay Estimator

The Driscoll-Kraay estimator provides robust covariance estimation for panel data with both cross-sectional and temporal dependence. It extends HAC estimation to panel data settings where observations are correlated both across time within units and potentially across units within time periods.

## Mathematical Foundation

### Panel Data Structure

For panel data with $N$ cross-sectional units observed over $T$ time periods, the Driscoll-Kraay estimator addresses the general correlation structure:

```math
\Omega = \sum_{h=-H}^H \sum_{s=-S}^S k_1\left(\frac{h}{H_T}\right) k_2\left(\frac{s}{S_T}\right) \Gamma_{h,s}
```

where:
- $h$ indexes time lags
- $s$ indexes cross-sectional lags
- $k_1(\cdot)$ and $k_2(\cdot)$ are kernel functions for time and cross-sectional dimensions
- $\Gamma_{h,s}$ are the cross-sectional and temporal autocovariances

### Key Features

1. **Dual Kernel Structure**: Uses separate kernels for time and spatial dimensions
2. **Non-parametric**: No assumptions about the specific correlation structure
3. **Asymptotic Consistency**: Consistent as both $N$ and $T$ grow large
4. **Automatic PSD**: Choice of kernels ensures positive semi-definiteness

## Core Type

```@docs
DriscollKraay
```

## Usage Examples

### Basic Panel Data

```julia
using CovarianceMatrices, DataFrames, GLM, Random
Random.seed!(123)

# Generate panel data
N_countries = 20
N_years = 15
N = N_countries * N_years

country_ids = repeat(1:N_countries, N_years)
year_ids = repeat(1:N_years, inner=N_countries)

# Generate data with both cross-sectional and temporal dependence
country_effects = randn(N_countries)[country_ids]
year_effects = randn(N_years)[year_ids]
X = randn(N, 2)

# Add spatial correlation (neighboring countries)
spatial_shock = randn(N_years)
ε = country_effects + year_effects +
    0.3 * spatial_shock[year_ids] .* sin.(2π * country_ids / N_countries) +
    randn(N) * 0.5

y = 2.0 .+ 1.2 * X[:, 1] - 0.8 * X[:, 2] + ε

df = DataFrame(
    y = y,
    x1 = X[:, 1],
    x2 = X[:, 2],
    country = country_ids,
    year = year_ids
)

# Fit model
model = lm(@formula(y ~ x1 + x2), df)

# Driscoll-Kraay with automatic bandwidth selection
dk_auto = DriscollKraay(Bartlett{Andrews}(), tis=df.year, iis=df.country)
se_dk = stderror(dk_auto, model)

# Compare with other approaches
se_classical = stderror(model)
se_cluster_country = stderror(CR1(df.country), model)
se_cluster_year = stderror(CR1(df.year), model)

results = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Classical = round.(se_classical, digits=4),
    Cluster_Country = round.(se_cluster_country, digits=4),
    Cluster_Year = round.(se_cluster_year, digits=4),
    Driscoll_Kraay = round.(se_dk, digits=4)
)

println(results)
```

### Different Kernel Specifications

```julia
# Various kernel combinations for different correlation patterns

# For strong temporal persistence, weaker spatial correlation
dk_bartlett_parzen = DriscollKraay(
    Bartlett{Andrews}(),  # Time dimension kernel
    tis = df.year,
    iis = df.country
)

# Fixed bandwidth for both dimensions
dk_fixed = DriscollKraay(
    Bartlett(4),  # Fixed bandwidth of 4
    tis = df.year,
    iis = df.country
)

# Compare different specifications
estimators = [
    ("Auto Bartlett", DriscollKraay(Bartlett{Andrews}(), tis=df.year, iis=df.country)),
    ("Fixed Bartlett", DriscollKraay(Bartlett(3), tis=df.year, iis=df.country)),
    ("Auto Parzen", DriscollKraay(Parzen{Andrews}(), tis=df.year, iis=df.country))
]

println("Kernel\\t\\tStd Errors")
for (name, est) in estimators
    se = stderror(est, model)
    println("$name\\t$(round.(se[2:3], digits=4))")
end
```

### Matrix Interface

```julia
# Direct matrix interface for residuals/moment conditions
residuals_matrix = residuals(model)

# Reshape to matrix format if needed
residuals_panel = reshape(residuals_matrix, :, 1)

# Apply Driscoll-Kraay directly
dk_matrix = DriscollKraay(Bartlett{Andrews}(), tis=df.year, iis=df.country)
Ω_dk = aVar(dk_matrix, residuals_panel)

println("Driscoll-Kraay covariance matrix:")
println(round.(Ω_dk, digits=6))
```

## When to Use Driscoll-Kraay

### ✅ Recommended for:

1. **Panel data with spatial correlation**:
   - Countries/regions with geographic proximity effects
   - Industry data with peer effects
   - Network data with spillover effects

2. **Long panels with serial correlation**:
   - Macro panels with persistent shocks
   - Financial panels with common market factors

3. **Balanced or mildly unbalanced panels**:
   - Works best with rectangular panel structure
   - Can handle some missing observations

### ❌ Consider alternatives for:

1. **Short panels (T < 10)**:
   - Use clustered standard errors instead
   - Insufficient time dimension for HAC benefits

2. **Severely unbalanced panels**:
   - Many missing observations can affect performance
   - Consider panel-specific methods

3. **Pure cross-sectional or time series data**:
   - Use HAC (time series) or HC (cross-sectional) instead

## Advanced Usage

### Diagnostic Tools

```julia
function panel_diagnostics(year_var, country_var)
    N = length(unique(country_var))
    T = length(unique(year_var))
    total_obs = length(year_var)

    balance = total_obs / (N * T)

    println("Panel Diagnostics:")
    println("Cross-sectional units (N): $N")
    println("Time periods (T): $T")
    println("Total observations: $total_obs")
    println("Balance ratio: $(round(balance, digits=3))")

    if balance < 0.8
        println("⚠️  Panel is unbalanced. Consider robustness checks.")
    end

    if T < 10
        println("⚠️  Short panel. HAC benefits may be limited.")
    end

    if N < 20
        println("⚠️  Few cross-sectional units. Spatial correlation hard to identify.")
    end

    return N, T, balance
end

panel_diagnostics(df.year, df.country)
```

### Bandwidth Selection Analysis

```julia
# Compare automatic vs fixed bandwidth selection
function bandwidth_comparison(df, model)
    # Get automatic bandwidth
    dk_auto = DriscollKraay(Bartlett{Andrews}(), tis=df.year, iis=df.country)
    se_auto = stderror(dk_auto, model)

    # Extract the bandwidth (this would require accessing internal state)
    # auto_bw = dk_auto.K.bw[1]  # Conceptual - actual access may differ

    # Try range of fixed bandwidths
    bandwidths = [1, 2, 3, 4, 5, 6]

    println("Bandwidth\\tStd Error (x1)\\tStd Error (x2)")
    println("Auto\\t\\t$(round(se_auto[2], digits=4))\\t\\t$(round(se_auto[3], digits=4))")

    for bw in bandwidths
        dk_fixed = DriscollKraay(Bartlett(bw), tis=df.year, iis=df.country)
        se_fixed = stderror(dk_fixed, model)
        println("$bw\\t\\t$(round(se_fixed[2], digits=4))\\t\\t$(round(se_fixed[3], digits=4))")
    end
end

bandwidth_comparison(df, model)
```

### Performance Considerations

```julia
using BenchmarkTools

function dk_performance_analysis()
    # Generate larger panel for timing
    N_large = 50
    T_large = 20
    N_obs = N_large * T_large

    large_years = repeat(1:T_large, inner=N_large)
    large_countries = repeat(1:N_large, T_large)
    large_residuals = randn(N_obs, 1)

    # Time different approaches
    dk_est = DriscollKraay(Bartlett(3), tis=large_years, iis=large_countries)

    time_dk = @belapsed aVar($dk_est, $large_residuals)

    # Compare with clustered alternatives
    cr_est = CR1(large_countries)
    time_cr = @belapsed aVar($cr_est, $large_residuals)

    println("Performance Comparison:")
    println("Driscoll-Kraay: $(round(time_dk * 1000, digits=2))ms")
    println("Clustered (CR1): $(round(time_cr * 1000, digits=2))ms")

    println("Driscoll-Kraay is $(round(time_dk/time_cr, digits=1))x slower than clustering")
end

dk_performance_analysis()
```

## Theoretical Background

### Asymptotic Properties

The Driscoll-Kraay estimator is consistent under the following conditions:

1. **Large N, T asymptotics**: Both cross-sectional and time dimensions grow
2. **Stationarity**: The panel is covariance stationary
3. **Weak dependence**: Correlations decay sufficiently fast

### Kernel Requirements

For positive semi-definiteness:
- Time kernel: Same as HAC kernels (Bartlett, Parzen, etc.)
- Spatial kernel: Must be positive semi-definite
- Common choice: Use same kernel for both dimensions

### Bandwidth Selection

The package implements Andrews-type bandwidth selection adapted for panels:
- Time bandwidth: Selected as in standard HAC
- Cross-sectional bandwidth: Often set to encompass relevant spatial correlation

## Integration with Other Methods

### Comparison Framework

```julia
function comprehensive_panel_comparison(df, model)
    methods = [
        ("Classical", nothing),
        ("Cluster Country", CR1(df.country)),
        ("Cluster Year", CR1(df.year)),
        ("Two-way Cluster", CR1((df.country, df.year))),
        ("Driscoll-Kraay", DriscollKraay(Bartlett{Andrews}(), tis=df.year, iis=df.country))
    ]

    results = DataFrame(Variable = ["Intercept", "x1", "x2"])

    for (method_name, estimator) in methods
        if method_name == "Classical"
            se = stderror(model)
        else
            se = stderror(estimator, model)
        end
        results[!, Symbol(method_name)] = round.(se, digits=4)
    end

    return results
end

full_comparison = comprehensive_panel_comparison(df, model)
println(full_comparison)
```

### Decision Tree

```julia
function panel_estimator_recommendation(N, T, balance_ratio)
    if T < 10
        return "Use clustered standard errors (CR1)"
    elseif N < 10
        return "Use time-series HAC (insufficient cross-section)"
    elseif balance_ratio < 0.5
        return "Consider panel-specific methods (severely unbalanced)"
    else
        if N < 30 && T < 30
            return "Use Driscoll-Kraay with caution (moderate sample)"
        else
            return "Driscoll-Kraay recommended"
        end
    end
end

# Apply to our data
N, T, balance = panel_diagnostics(df.year, df.country)
recommendation = panel_estimator_recommendation(N, T, balance)
println("\\nRecommendation: $recommendation")
```

## Best Practices

### Model Specification

1. **Check panel balance**: Severely unbalanced panels may cause issues
2. **Examine residual patterns**: Look for evidence of spatial/temporal correlation
3. **Consider fixed effects**: May absorb some correlation structures

### Bandwidth Selection

1. **Start with automatic**: Use Andrews selection as baseline
2. **Sensitivity analysis**: Try range of fixed bandwidths
3. **Cross-validation**: Use out-of-sample performance if available

### Interpretation

1. **Conservative inference**: DK standard errors tend to be larger
2. **Asymptotic theory**: Requires both N and T reasonably large
3. **Robustness**: More robust than clustering to correlation patterns

## Common Applications

### Macroeconomic Panels

```julia
# Country-level data with spillover effects
# Growth, trade, financial integration, etc.
```

### Financial Panels

```julia
# Firm-level data with common market factors
# Returns, performance, risk measures, etc.
```

### Environmental/Spatial Data

```julia
# Geographic data with spatial correlation
# Pollution, climate, resource extraction, etc.
```

## References

**Original Paper:**
- Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data". *Review of Economics and Statistics*, 80(4), 549-560.

**Methodological Extensions:**
- Hoechle, D. (2007). "Robust standard errors for panel regressions with cross-sectional dependence". *The Stata Journal*, 7(3), 281-312.
- Vogelsang, T.J. (2012). "Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects". *Journal of Econometrics*, 166(2), 303-319.

**Applications:**
- Beck, N. and Katz, J.N. (1995). "What to do (and not to do) with time-series cross-section data". *American Political Science Review*, 89(3), 634-647.