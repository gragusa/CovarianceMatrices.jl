# GLM Integration Tutorial

This tutorial demonstrates how to use CovarianceMatrices.jl with GLM.jl for robust inference in regression models. The integration provides seamless robust standard errors for linear and generalized linear models.

## Overview

CovarianceMatrices.jl extends GLM.jl's `vcov()` and `stderror()` functions with robust covariance estimators. The general workflow is:

1. Fit your model using GLM.jl (`lm()`, `glm()`, etc.)
2. Compute robust covariance matrix using `vcov(estimator, model)`
3. Extract robust standard errors using `stderror(estimator, model)`
4. Perform inference using robust statistics

```julia
using GLM, CovarianceMatrices, DataFrames, Random, Statistics
using LinearAlgebra, StatsBase, Distributions
Random.seed!(123)
```

## Example 1: Linear Regression with Heteroskedasticity

Let's start with a simple linear regression that exhibits heteroskedasticity:

```julia
# Generate data with heteroskedasticity
n = 200
x1 = randn(n)
x2 = randn(n)
x3 = randn(n)

# Heteroskedastic error variance depends on x1
σ² = exp.(0.5 .+ 0.8 * abs.(x1))
ε = sqrt.(σ²) .* randn(n)

# True coefficients
β = [2.0, 1.5, -1.0, 0.5]
y = β[1] .+ β[2] * x1 .+ β[3] * x2 .+ β[4] * x3 .+ ε

# Create DataFrame
df = DataFrame(y=y, x1=x1, x2=x2, x3=x3)

# Fit OLS model
model = lm(@formula(y ~ x1 + x2 + x3), df)
println("R² = $(round(r2(model), digits=3))")
println("True coefficients: $β")
println("OLS estimates: $(round.(coef(model), digits=3))")
```

### Comparing Standard Errors

```julia
# Classical standard errors (assume homoskedasticity)
se_classical = stderror(model)

# Robust standard errors (HC estimators)
se_hc0 = stderror(HC0(), model)
se_hc1 = stderror(HC1(), model)
se_hc2 = stderror(HC2(), model)
se_hc3 = stderror(HC3(), model)  # Most common choice

# Display results
results_df = DataFrame(
    Variable = ["Intercept", "x1", "x2", "x3"],
    Coefficient = round.(coef(model), digits=3),
    SE_Classical = round.(se_classical, digits=3),
    SE_HC0 = round.(se_hc0, digits=3),
    SE_HC1 = round.(se_hc1, digits=3),
    SE_HC2 = round.(se_hc2, digits=3),
    SE_HC3 = round.(se_hc3, digits=3)
)

println("\nStandard Error Comparison:")
println(results_df)
```

### Variance-Covariance Matrices

```julia
# Compare covariance matrices
vcov_classical = vcov(model)
vcov_hc3 = vcov(HC3(), model)

println("\nDiagonal ratio (HC3/Classical):")
for i in 1:4
    ratio = vcov_hc3[i,i] / vcov_classical[i,i]
    println("  Variable $(i): $(round(ratio, digits=2))")
end
```

## Example 2: Time Series Regression with Autocorrelation

Now let's consider a time series regression with serial correlation:

```julia
# Generate time series data
T = 300
t = 1:T

# Trending regressors
trend = collect(t) ./ T
seasonal = sin.(2π * t / 12)  # Monthly seasonality
x1_ts = trend + 0.3 * randn(T)
x2_ts = seasonal + 0.2 * randn(T)

# AR(1) errors
ρ = 0.6
ε_ts = zeros(T)
ε_ts[1] = randn()
for i in 2:T
    ε_ts[i] = ρ * ε_ts[i-1] + randn()
end

# Generate dependent variable
β_ts = [1.0, 2.0, 1.5]
y_ts = β_ts[1] .+ β_ts[2] * x1_ts .+ β_ts[3] * x2_ts .+ ε_ts

# Create time series DataFrame
df_ts = DataFrame(y=y_ts, x1=x1_ts, x2=x2_ts, t=t)

# Fit time series model
model_ts = lm(@formula(y ~ x1 + x2), df_ts)
println("\nTime Series Model R² = $(round(r2(model_ts), digits=3))")
```

### HAC Standard Errors

```julia
# Classical (incorrect for time series)
se_ts_classical = stderror(model_ts)

# HAC standard errors
se_bartlett_andrews = stderror(Bartlett{Andrews}(), model_ts)
se_bartlett_fixed = stderror(Bartlett(6), model_ts)
se_parzen_nw = stderror(Parzen{NeweyWest}(), model_ts)
se_qs = stderror(QuadraticSpectral{Andrews}(), model_ts)

# VARHAC (automatic, no bandwidth selection)
se_varhac = stderror(VARHAC(), model_ts)

# Smoothed Moments
se_smoothed = stderror(SmoothedMoments(), model_ts)

# Results comparison
results_ts = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Coefficient = round.(coef(model_ts), digits=3),
    SE_Classical = round.(se_ts_classical, digits=3),
    SE_Bartlett = round.(se_bartlett_andrews, digits=3),
    SE_Parzen = round.(se_parzen_nw, digits=3),
    SE_VARHAC = round.(se_varhac, digits=3),
    SE_Smoothed = round.(se_smoothed, digits=3)
)

println("\nTime Series Standard Error Comparison:")
println(results_ts)
```

### Bandwidth Diagnosis

```julia
# Check bandwidth selection
_, _, bw_bartlett = workingoptimalbw(Bartlett{Andrews}(), residuals(model_ts)')
_, _, bw_parzen = workingoptimalbw(Parzen{NeweyWest}(), residuals(model_ts)')

println("\nBandwidth Selection:")
println("  Bartlett (Andrews): $(round(bw_bartlett, digits=2))")
println("  Parzen (Newey-West): $(round(bw_parzen, digits=2))")

# VARHAC lag selection
varhac_est = VARHAC()
_ = vcov(varhac_est, model_ts)  # Fit the model
println("  VARHAC selected lags: $(order(varhac_est))")
```

## Example 3: Panel Data and Clustered Standard Errors

Panel data often requires clustered standard errors:

```julia
# Generate panel data
n_firms = 50
n_years = 8
n_panel = n_firms * n_years

# Panel identifiers
firm_id = repeat(1:n_firms, inner=n_years)
year_id = repeat(1:n_years, outer=n_firms)

# Firm fixed effects
firm_effects = randn(n_firms)[firm_id] * 2.0
# Year fixed effects
year_effects = randn(n_years)[year_id] * 1.0

# Regressors
x_panel = randn(n_panel)
# Panel error with firm-level clustering
firm_shocks = randn(n_firms)[firm_id] * 1.5
idiosyncratic = randn(n_panel) * 0.8
ε_panel = firm_shocks + idiosyncratic

# Dependent variable
y_panel = 1.0 .+ 0.8 * x_panel .+ firm_effects .+ year_effects .+ ε_panel

# Create panel DataFrame
df_panel = DataFrame(
    y = y_panel,
    x = x_panel,
    firm_id = firm_id,
    year_id = year_id
)

# Fit panel model (without fixed effects for simplicity)
model_panel = lm(@formula(y ~ x), df_panel)
println("\nPanel Model R² = $(round(r2(model_panel), digits=3))")
```

### Clustered Standard Errors

```julia
# Standard errors
se_panel_classical = stderror(model_panel)

# Firm-clustered standard errors
se_cr0_firm = stderror(CR0(firm_id), model_panel)
se_cr1_firm = stderror(CR1(firm_id), model_panel)
se_cr2_firm = stderror(CR2(firm_id), model_panel)
se_cr3_firm = stderror(CR3(firm_id), model_panel)

# Two-way clustering (firm and year)
se_cr1_twoway = stderror(CR1((firm_id, year_id)), model_panel)

# Driscoll-Kraay (spatial-temporal)
se_dk = stderror(DriscollKraay(Bartlett{Andrews}(), tis=year_id, iis=firm_id), model_panel)

# Panel results
results_panel = DataFrame(
    Variable = ["Intercept", "x"],
    Coefficient = round.(coef(model_panel), digits=3),
    SE_Classical = round.(se_panel_classical, digits=3),
    SE_CR0_Firm = round.(se_cr0_firm, digits=3),
    SE_CR1_Firm = round.(se_cr1_firm, digits=3),
    SE_CR2_Firm = round.(se_cr2_firm, digits=3),
    SE_TwoWay = round.(se_cr1_twoway, digits=3),
    SE_DriscollKraay = round.(se_dk, digits=3)
)

println("\nPanel Standard Error Comparison:")
println(results_panel)
```

## Example 4: Logistic Regression

CovarianceMatrices.jl also works with generalized linear models:

```julia
# Generate binary choice data
n_logit = 500
x1_logit = randn(n_logit)
x2_logit = randn(n_logit)

# Logit model: heteroskedasticity is inherent
β_logit = [0.5, 1.2, -0.8]
linear_pred = β_logit[1] .+ β_logit[2] * x1_logit .+ β_logit[3] * x2_logit
prob = 1 ./ (1 .+ exp.(-linear_pred))
y_binary = rand.(Bernoulli.(prob))

df_logit = DataFrame(y=y_binary, x1=x1_logit, x2=x2_logit)

# Fit logistic regression
model_logit = glm(@formula(y ~ x1 + x2), df_logit, Binomial(), LogitLink())
println("\nLogistic Regression Deviance = $(round(deviance(model_logit), digits=2))")
```

### Robust Standard Errors for GLM

```julia
# Standard errors for logit model
se_logit_classical = stderror(model_logit)
se_logit_hc3 = stderror(HC3(), model_logit)

# For clustered data (create artificial clusters)
clusters_logit = repeat(1:25, inner=20)
se_logit_cr1 = stderror(CR1(clusters_logit), model_logit)

results_logit = DataFrame(
    Variable = ["Intercept", "x1", "x2"],
    Coefficient = round.(coef(model_logit), digits=3),
    SE_Classical = round.(se_logit_classical, digits=3),
    SE_HC3 = round.(se_logit_hc3, digits=3),
    SE_CR1 = round.(se_logit_cr1, digits=3)
)

println("\nLogistic Regression Standard Error Comparison:")
println(results_logit)
```

## Example 5: Advanced Usage with New API

The package includes a new unified API that provides additional flexibility:

```julia
# Using the new variance forms API
using CovarianceMatrices: Information, Misspecified

# Information matrix equality (assumes correct specification)
vcov_info = vcov(HC3(), Information(), model)
se_info = sqrt.(diag(vcov_info))

# Robust sandwich form (allows misspecification)
vcov_robust = vcov(HC3(), Misspecified(), model)
se_robust = sqrt.(diag(vcov_robust))

println("\nVariance Forms Comparison:")
println("Information form SEs: $(round.(se_info, digits=3))")
println("Robust sandwich SEs: $(round.(se_robust, digits=3))")
```

## Practical Guidelines

### Choosing the Right Estimator

#### For Cross-Sectional Data:
```julia
# Small samples (n < 250): HC2 or HC3
se_small_sample = stderror(HC3(), model)

# Large samples: HC0 or HC1 acceptable
se_large_sample = stderror(HC0(), model)

# When in doubt: HC3 is generally robust choice
se_recommended = stderror(HC3(), model)
```

#### For Time Series Data:
```julia
# Conservative approach: Bartlett with Andrews bandwidth
se_conservative = stderror(Bartlett{Andrews}(), model_ts)

# Automatic approach: VARHAC
se_automatic = stderror(VARHAC(), model_ts)

# When bandwidth matters: Try different kernels
se_parzen = stderror(Parzen{Andrews}(), model_ts)
se_qs = stderror(QuadraticSpectral{Andrews}(), model_ts)
```

#### For Panel Data:
```julia
# Firm clustering
se_firm_cluster = stderror(CR1(firm_id), model_panel)

# Two-way clustering
se_twoway = stderror(CR1((firm_id, year_id)), model_panel)

# Spatial-temporal correlation
se_spatial_temporal = stderror(DriscollKraay(Bartlett{Andrews}(),
                                          tis=year_id, iis=firm_id), model_panel)
```

### Diagnostic and Sensitivity Analysis

```julia
# Function to compare multiple estimators
function robust_comparison(model, estimators, names)
    results = DataFrame(Variable = ["Intercept", "x1", "x2", "x3"][1:length(coef(model))])
    results.Coefficient = round.(coef(model), digits=3)

    for (est, name) in zip(estimators, names)
        results[!, Symbol("SE_" * name)] = round.(stderror(est, model), digits=3)
    end

    return results
end

# Apply to our heteroskedastic example
estimators = [HC0(), HC1(), HC2(), HC3(), Bartlett(5), VARHAC()]
names = ["HC0", "HC1", "HC2", "HC3", "HAC", "VARHAC"]

comparison_results = robust_comparison(model, estimators, names)
println("\nComprehensive Comparison:")
println(comparison_results)
```

### Testing for Specification Issues

```julia
# Simple heteroskedasticity test using residual patterns
residuals_sq = residuals(model).^2
het_test_model = lm(Term(:residuals_sq) ~ sum(term.([:x1, :x2, :x3])),
                   DataFrame(residuals_sq=residuals_sq, x1=x1, x2=x2, x3=x3))
het_test_stat = r2(het_test_model) * length(residuals_sq)
het_p_value = 1 - cdf(Chisq(3), het_test_stat)

println("\nBreusch-Pagan Test for Heteroskedasticity:")
println("  Test statistic: $(round(het_test_stat, digits=2))")
println("  p-value: $(round(het_p_value, digits=4))")
if het_p_value < 0.05
    println("  Conclusion: Reject homoskedasticity (use robust SEs)")
else
    println("  Conclusion: Fail to reject homoskedasticity")
end
```

## Summary and Best Practices

### Default Recommendations:

1. **Cross-sectional data**: Use `HC3()` as default robust estimator
2. **Time series data**: Use `VARHAC()` for automatic approach, `Bartlett{Andrews}()` for traditional HAC
3. **Panel data**: Use `CR1()` for one-way clustering, `CR1((cluster1, cluster2))` for two-way
4. **Mixed cases**: Start with `VARHAC()` or `SmoothedMoments()` for guaranteed positive definiteness

### Performance Considerations:

```julia
# Quick performance comparison
using BenchmarkTools

println("\nPerformance comparison (on fitted model):")
@btime vcov(HC3(), $model)
@btime vcov(VARHAC(), $model_ts)
@btime vcov(CR1($firm_id), $model_panel)
@btime vcov(Bartlett{Andrews}(), $model_ts)
```

### Integration with Other Packages:

The robust standard errors computed here integrate seamlessly with:
- **StatsBase.jl**: `confint()`, `coeftable()`
- **GLM.jl**: All standard model methods
- **FixedEffectModels.jl**: Panel data with fixed effects
- **MixedModels.jl**: Random effects models

This completes the GLM integration tutorial. The combination of CovarianceMatrices.jl with GLM.jl provides a powerful and flexible framework for robust inference in econometric applications.