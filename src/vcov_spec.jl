##############################################################################
##
## VcovSpec - Wrapper for variance-covariance specifications
##
## Enables `model + vcov(estimator)` syntax for robust standard errors.
##
##############################################################################

"""
    VcovSpec{T}

Wrapper for variance-covariance specifications.
Enables `model + vcov(estimator)` syntax.

# Examples
```julia
# Create VcovSpec for heteroskedasticity-robust inference
v = vcov(HC3())

# Use with + operator (requires package-specific implementation)
model = lm(@formula(y ~ x), df)
model_hc3 = model + vcov(HC3())
```
"""
struct VcovSpec{T}
    source::T
end

Base.show(io::IO, v::VcovSpec) = print(io, "VcovSpec(", v.source, ")")
Base.show(io::IO, ::MIME"text/plain", v::VcovSpec) = print(io, "VcovSpec wrapping: ", v.source)

"""
    vcov(estimator::AbstractAsymptoticVarianceEstimator) -> VcovSpec

Create a VcovSpec for use with `model + vcov(...)` syntax.

This single-argument form wraps the variance estimator in a VcovSpec,
which can then be added to a fitted model using the `+` operator.

# Arguments
- `estimator`: A variance estimator (HC0, HC1, CR1, etc.)

# Returns
- `VcovSpec{V}`: A wrapper containing the estimator

# Examples
```julia
# Create VcovSpec for heteroskedasticity-robust inference
v = vcov(HC3())

# Use with + operator
model = ols(df, @formula(y ~ x))
model_hc3 = model + vcov(HC3())

# Cluster-robust
model_cr1 = model + vcov(CR1(:firm))
```
"""
StatsAPI.vcov(v::AbstractAsymptoticVarianceEstimator) = VcovSpec(v)

# Idempotent: vcov(vcov(x)) == vcov(x)
StatsAPI.vcov(v::VcovSpec) = v
