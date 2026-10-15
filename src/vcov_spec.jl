##############################################################################
##
## VcovSpec - Wrapper for variance-covariance specifications
##
## Enables `model + vcov(estimator)` syntax for robust standard errors.
##
##############################################################################

"""
    VcovSpec{T}

A variance estimator packaged for a modeling package, built by the one-argument
`vcov(estimator)`. The `source` field holds the estimator unchanged.

CovarianceMatrices defines no operation that consumes a `VcovSpec`. A modeling package
that supports the `model + vcov(estimator)` idiom defines `Base.:+(m::MyModel,
v::VcovSpec)`, reads the estimator from `v.source`, and returns the model with that
estimator attached. If the estimator names its clusters by column (`CR1(:firm)`), the
modeling package replaces the names with the group vectors from its data before
computing the variance; see the "For Modeling Packages" section of the interface
tutorial.

# Examples
```julia
spec = vcov(CR1(:firm))   # VcovSpec(CR1(:firm))
spec.source               # CR1(:firm)
```
"""
struct VcovSpec{T}
    source::T
end

Base.show(io::IO, v::VcovSpec) = print(io, "VcovSpec(", v.source, ")")
function Base.show(io::IO, ::MIME"text/plain", v::VcovSpec)
    print(io, "VcovSpec wrapping: ", v.source)
end

"""
    vcov(estimator::AbstractAsymptoticVarianceEstimator) -> VcovSpec

Wrap `estimator` in a [`VcovSpec`](@ref) without computing anything. Modeling
packages consume the result through `model + vcov(estimator)`; this package does not
define that operator.

`vcov(vcov(estimator)) == vcov(estimator)`.

# Examples
```julia
vcov(HC3())            # VcovSpec(HC3())
vcov(CR1(:firm))       # cluster names are resolved by the modeling package
```
"""
StatsAPI.vcov(v::AbstractAsymptoticVarianceEstimator) = VcovSpec(v)

StatsAPI.vcov(v::VcovSpec) = v
