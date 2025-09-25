# Heteroskedasticity-Robust (HC/HR) Estimators

HC (Heteroskedasticity-Consistent) estimators, also known as HR (Heteroskedasticity-Robust) estimators, provide robust covariance matrices for cross-sectional data exhibiting conditional heteroskedasticity.

## Mathematical Foundation

For cross-sectional data where $\mathbb{E}[\varepsilon_t^2 | X_t] = \sigma_t^2$ (heteroskedasticity) but $\mathbb{E}[\varepsilon_t \varepsilon_s | X_t, X_s] = 0$ for $t \neq s$ (no serial correlation), the robust covariance matrix is:

```math
\hat{\Omega}_{HC} = \frac{1}{T} \sum_{t=1}^T \phi_j(h_t) g_t g_t'
```

where $\phi_j(h_t)$ are adjustment factors that depend on leverage values $h_t$ and the specific HC variant.

## Available Estimators

```@docs
HC0
HC1
HC2
HC3
HC4
HC5
```

### HC0 (White's Original Estimator)

```math
\phi_0(h_t) = 1
```

**Properties:**
- Original White (1980) estimator
- Simplest form
- Can be severely biased in small samples

**Usage:**
```julia
using CovarianceMatrices
hc0 = HC0()
Ω = aVar(hc0, residuals)
```

### HC1 (Degrees of Freedom Correction)

```math
\phi_1(h_t) = \frac{T}{T-k}
```

**Properties:**
- Applies degrees of freedom correction
- Reduces bias compared to HC0
- Still can be problematic with high leverage points

### HC2 (Leverage-Based Correction)

```math
\phi_2(h_t) = \frac{1}{1-h_t}
```

**Properties:**
- Adjusts for leverage directly
- Better performance with influential observations
- Unbiased under homoskedasticity

### HC3 (Preferred for Small Samples)

```math
\phi_3(h_t) = \frac{1}{(1-h_t)^2}
```

**Properties:**
- Most commonly recommended
- Good finite sample properties
- Robust to influential observations
- **Default choice for most applications**

### HC4 and HC5 (Advanced Corrections)

**HC4:**
```math
\phi_4(h_t) = \frac{1}{(1-h_t)^{\delta_t}}
```
where $\delta_t = \min(4, T \cdot h_t / \bar{h})$

**HC5:**
```math
\phi_5(h_t) = \frac{1}{(1-h_t)^{\gamma_t}}
```
where $\gamma_t = \min(\max(4, 0.7 \cdot \max(2, T \cdot h_t / k)), 0.7 \cdot \max(1.5, T \cdot h_t / k))$

**Properties:**
- Designed for extreme leverage cases
- More complex adjustments
- Use when standard HC variants perform poorly

## Practical Usage

### Basic Cross-Sectional Example

```julia
using CovarianceMatrices, Random, LinearAlgebra
Random.seed!(123)

# Generate heteroskedastic data
n = 200
X = [ones(n) randn(n, 2)]  # Design matrix with intercept
β = [1.0, 0.5, -0.3]

# Heteroskedastic errors (variance depends on X)
σ² = exp.(0.5 .+ 0.8 * abs.(X[:, 2]))
ε = sqrt.(σ²) .* randn(n)
y = X * β + ε

# Compute residuals (these would be your moment conditions)
residuals = reshape(y - X * β, n, 1)

# Compare all HC estimators
hc_estimators = [HC0(), HC1(), HC2(), HC3(), HC4(), HC5()]
hc_names = ["HC0", "HC1", "HC2", "HC3", "HC4", "HC5"]

println("Estimator\tVariance\tStd Error")
for (est, name) in zip(hc_estimators, hc_names)
    Ω = aVar(est, residuals)
    variance = Ω[1, 1]
    se = sqrt(variance)
    println("$name\t\t$(round(variance, digits=4))\t$(round(se, digits=4))")
end
```

### GLM Integration

```julia
using GLM, DataFrames

# Create DataFrame
df = DataFrame(y=y, x1=X[:, 2], x2=X[:, 3])

# Fit regression model
model = lm(@formula(y ~ x1 + x2), df)

# Compare standard errors
se_classical = stderror(model)
se_hc3 = stderror(HC3(), model)

results = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Classical_SE = round.(se_classical, digits=4),
    HC3_SE = round.(se_hc3, digits=4),
    Ratio = round.(se_hc3 ./ se_classical, digits=2)
)

println(results)
```

## Selection Guidelines

### Sample Size Considerations

```julia
function hc_sample_size_analysis()
    sample_sizes = [50, 100, 200, 500, 1000]

    for n in sample_sizes
        # Generate data
        X_sim = [ones(n) randn(n)]
        σ²_sim = exp.(0.5 * abs.(X_sim[:, 2]))
        ε_sim = sqrt.(σ²_sim) .* randn(n)
        residuals_sim = reshape(ε_sim, n, 1)

        # Compare HC2 and HC3
        Ω_hc2 = aVar(HC2(), residuals_sim)
        Ω_hc3 = aVar(HC3(), residuals_sim)

        ratio = Ω_hc3[1,1] / Ω_hc2[1,1]

        println("n=$n: HC3/HC2 ratio = $(round(ratio, digits=2))")
    end
end

hc_sample_size_analysis()
```

### Leverage Diagnostics

```julia
using GLM

function leverage_diagnostics(model)
    X = modelmatrix(model)
    H = X * inv(X' * X) * X'  # Hat matrix
    leverages = diag(H)

    println("Leverage diagnostics:")
    println("Mean leverage: $(round(mean(leverages), digits=4))")
    println("Max leverage: $(round(maximum(leverages), digits=4))")
    println("Number of high leverage (>2k/n): $(sum(leverages .> 2*size(X,2)/size(X,1)))")

    # High leverage threshold
    k, n = size(X, 2), size(X, 1)
    high_leverage = leverages .> 2*k/n

    if any(high_leverage)
        println("Recommendation: Use HC3, HC4, or HC5")
        return "HC3"  # Conservative choice
    else
        println("Recommendation: HC1 or HC2 sufficient")
        return "HC1"
    end
end

# Apply to model
rec = leverage_diagnostics(model)
```

## Advanced Usage

### Comparing All Estimators Systematically

```julia
function comprehensive_hc_comparison(model; include_classical=true)
    estimators = [HC0(), HC1(), HC2(), HC3(), HC4(), HC5()]
    names = ["HC0", "HC1", "HC2", "HC3", "HC4", "HC5"]

    if include_classical
        insert!(names, 1, "Classical")
    end

    # Get coefficients and their count
    coeffs = coef(model)
    k = length(coeffs)

    # Results matrix
    results = DataFrame(Variable = ["Intercept"; ["x$i" for i in 1:(k-1)]])

    # Classical standard errors
    if include_classical
        results.Classical = round.(stderror(model), digits=4)
    end

    # HC standard errors
    for (est, name) in zip(estimators, names[include_classical ? 2:end : 1:end])
        se_robust = stderror(est, model)
        results[!, Symbol(name)] = round.(se_robust, digits=4)
    end

    return results
end

comp_results = comprehensive_hc_comparison(model)
println(comp_results)
```

### Testing for Heteroskedasticity

```julia
function breusch_pagan_test(model)
    # Compute squared residuals
    residuals_sq = residuals(model).^2

    # Get design matrix (excluding intercept for simplicity)
    X = modelmatrix(model)[:, 2:end]

    # Auxiliary regression
    aux_data = DataFrame(res_sq = residuals_sq)
    for i in 1:size(X, 2)
        aux_data[!, Symbol("x$i")] = X[:, i]
    end

    # Fit auxiliary regression
    formula_str = "res_sq ~ " * join(["x$i" for i in 1:size(X, 2)], " + ")
    aux_model = lm(eval(Meta.parse("@formula($formula_str)")), aux_data)

    # Test statistic
    n = length(residuals_sq)
    test_stat = n * r2(aux_model)
    df = size(X, 2)

    # p-value (Chi-squared distribution)
    using Distributions
    p_value = 1 - cdf(Chisq(df), test_stat)

    println("Breusch-Pagan Test for Heteroskedasticity:")
    println("Test statistic: $(round(test_stat, digits=3))")
    println("p-value: $(round(p_value, digits=4))")

    if p_value < 0.05
        println("Conclusion: Reject homoskedasticity → Use robust standard errors")
        return true  # Use robust SEs
    else
        println("Conclusion: Fail to reject homoskedasticity → Classical SEs may be adequate")
        return false  # Classical SEs okay
    end
end

use_robust = breusch_pagan_test(model)
```

## Performance Considerations

```julia
using BenchmarkTools

function hc_performance_comparison(n_values)
    println("Performance Comparison (time per computation):")
    println("n\tHC0\tHC1\tHC2\tHC3\tHC4\tHC5")

    for n in n_values
        # Generate test data
        X_perf = randn(n, 3)
        residuals_perf = randn(n, 1)

        # Benchmark each estimator
        times = Float64[]
        for est in [HC0(), HC1(), HC2(), HC3(), HC4(), HC5()]
            t = @belapsed aVar($est, $residuals_perf)
            push!(times, t * 1000)  # Convert to milliseconds
        end

        println("$n\t" * join([round(t, digits=2) for t in times], "\t"))
    end
end

hc_performance_comparison([100, 500, 1000, 5000])
```

## Alternative Names (HR)

The package also provides HC estimators under the HR (Heteroskedasticity-Robust) names:

```@docs
HR0
HR1
HR2
HR3
HR4
HR5
```

These are identical to their HC counterparts:

```julia
# These are equivalent
hc3 = HC3()
hr3 = HR3()

# Both produce identical results
Ω_hc = aVar(hc3, residuals)
Ω_hr = aVar(hr3, residuals)

println("Matrices identical: ", Ω_hc ≈ Ω_hr)  # Should print true
```

## Integration with Other Estimators

### When Not to Use HC Estimators

HC estimators are inappropriate for:

1. **Time series data** (use HAC estimators instead)
2. **Panel data with clustering** (use CR estimators)
3. **Spatial correlation** (use Driscoll-Kraay)

```julia
# Example: Detecting time series structure
function detect_autocorrelation(residuals_vec, max_lag=5)
    using StatsBase

    autocorrs = [cor(residuals_vec[1:end-lag], residuals_vec[lag+1:end])
                 for lag in 1:max_lag]

    println("Autocorrelations:")
    for (lag, ac) in enumerate(autocorrs)
        println("Lag $lag: $(round(ac, digits=3))")
    end

    significant = any(abs.(autocorrs) .> 0.1)  # Rule of thumb

    if significant
        println("Warning: Evidence of serial correlation → Consider HAC estimators")
    else
        println("No strong evidence of serial correlation → HC estimators appropriate")
    end

    return significant
end

# Test on model residuals
has_autocorr = detect_autocorrelation(residuals(model))
```

## Best Practices Summary

### Default Recommendations

1. **Standard choice**: Use `HC3()` for most applications
2. **Small samples (n < 250)**: Definitely use `HC3()`
3. **Large samples (n > 1000)**: `HC0()` or `HC1()` often sufficient
4. **High leverage concerns**: Consider `HC4()` or `HC5()`
5. **When in doubt**: `HC3()` is rarely a poor choice

### Workflow Recommendation

```julia
function robust_workflow(model)
    # Step 1: Test for heteroskedasticity
    needs_robust = breusch_pagan_test(model)

    if !needs_robust
        println("Using classical standard errors")
        return stderror(model)
    end

    # Step 2: Check for problematic leverage
    rec = leverage_diagnostics(model)

    # Step 3: Apply recommended estimator
    if rec == "HC3"
        println("Using HC3 standard errors")
        return stderror(HC3(), model)
    else
        println("Using HC1 standard errors")
        return stderror(HC1(), model)
    end
end

# Apply workflow
se_robust = robust_workflow(model)
```

HC/HR estimators provide the foundation for robust inference in cross-sectional econometric models and remain one of the most widely used tools in applied econometrics.

## References

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". *Econometrica*, 48(4), 817-838.
- MacKinnon, J.G. and White, H. (1985). "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties". *Journal of Econometrics*, 29(3), 305-325.
- Cribari-Neto, F. (2004). "Asymptotic inference under heteroskedasticity of unknown form". *Computational Statistics & Data Analysis*, 45(2), 215-233.