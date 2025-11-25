"""
Model integration interface for third-party estimators.

This module defines the duck-typed interface that third-party estimator
objects should implement to work with the CovarianceMatrices.jl API.
"""

"""
`MLikeModel`

Abstract type for Maximum Likelihood-like models.

Represents models that can be estimated using maximum likelihood methods
and have associated score functions and Hessian matrices.
"""
abstract type MLikeModel <: StatsAPI.StatisticalModel end

"""
`GMMLikeModel`

Abstract type for Generalized Method of Moments-like models.

Represents models that can be estimated using GMM methods and have
associated moment conditions.
"""
abstract type GMMLikeModel <: StatsAPI.StatisticalModel end

# Forward declarations for variance forms (defined in variance_forms.jl)
"""
`VarianceForm`

Abstract type for different variance matrix forms in robust covariance estimation.

Subtypes specify how the covariance matrix should be computed for different model assumptions:
- `Information`: Uses Fisher Information matrix (assumes correct specification)
- `Misspecified`: Uses sandwich/robust form (allows model misspecification)
"""
abstract type VarianceForm end

"""
`Information`

Variance form that uses the Fisher Information matrix.

Assumes the model is correctly specified. For MLE, this gives the Cramér-Rao lower bound.
For correctly specified models, this is more efficient than the sandwich form.

# Usage
```julia
using CovarianceMatrices
vcov_info = vcov(HC3(), Information(), model)
```
"""
struct Information <: VarianceForm end

"""
`Misspecified`

Variance form that uses the sandwich/robust estimator.

Allows for model misspecification. This is the standard "robust" covariance matrix
that remains valid even if the model is misspecified.

# Usage
```julia
using CovarianceMatrices
vcov_robust = vcov(HC3(), Misspecified(), model)
```
"""
struct Misspecified <: VarianceForm end

# Convenience type unions for dispatch
const MLikeForm = Union{Information, Misspecified}
const GMMLikeForm = Union{Information, Misspecified}

"""
    momentmatrix(model) -> AbstractMatrix
    momentmatrix(model, θ::AbstractVector) -> AbstractMatrix

Return the moment matrix for the estimation problem.

For MLE, this corresponds to the score functions evaluated at each observation.
For GMM, this represents the moment conditions evaluated at the observed data.

# Returns
- `AbstractMatrix`: n × m matrix where n is the number of observations
  and m is the number of moment conditions.

# Extended Interface
Models can optionally implement:
- `momentmatrix(model, θ)`: Moment matrix evaluated at parameter vector θ
"""
function momentmatrix end

# Note: momentmatrix is already defined in api.jl, we just extend the documentation here

"""
    cross_score(model) -> AbstractMatrix

Return the cross-product matrix of the moment conditions (also known as the "meat"
of the sandwich estimator or the Jacobian matrix of moment conditions).

This is the **unscaled** cross-product: G = ∑ᵢ gᵢgᵢ' where gᵢ are the moment
contributions (or equivalently g'g where g is the moment matrix).

# Returns
- `AbstractMatrix`: m × m matrix where m is the number of moment conditions.

```julia
function cross_score(model)
    g = momentmatrix(model)
    return (g' * g)
end
```

This default works for most cases. Override only if you have an analytical
expression or need custom behavior.

# Important: Scaling Convention
The returned matrix should be **unscaled** - that is, it should be the **sum**
of cross-products, NOT divided by the number of observations.

This convention ensures consistency with the unscaled Hessian and allows the
variance computation formulas to work correctly.

# Note
For MLE models, this is related to the Fisher Information matrix.
For GMM models, this is the "G" matrix in the variance formula.
"""
function cross_score(model)
    # Default implementation: unscaled cross-product of moment matrix
    g = momentmatrix(model)
    return (g' * g)
end

"""
    jacobian_momentfunction(model) -> AbstractMatrix

Return the Jacobian matrix of the moment function with respect to parameters.

This is the matrix of derivatives of the (summed) moment conditions with respect
to the parameters: ∂(∑ᵢ gᵢ)/∂θ' where gᵢ are individual moment contributions.

# Returns
- `AbstractMatrix`: m × k matrix where m is the number of moment conditions
  and k is the number of parameters.

# Important: Scaling Convention
The returned matrix should be **unscaled** - the derivative of the sum of moments,
not the average.

# Note
For GMM models, this is required for variance computation.
For MLE models, this may equal the negative cross_score matrix.
"""
function jacobian_momentfunction(model)
    # Default: not available
    return nothing
end

"""
    hessian_objective(model) -> Union{Nothing, AbstractMatrix}

Return the Hessian matrix of the estimator's objective function.

This is the matrix of second derivatives of the objective function with
respect to the parameters, evaluated at the estimated parameters.

# Returns
- `AbstractMatrix`: k × k matrix where k is the number of parameters
- `Nothing`: if not available or not applicable

# Important: Scaling Convention
The returned matrix should be **unscaled**, that is, NOT divided by the number of observations

```

# Note
For MLE, this is the negative Hessian of the log-likelihood.

"""
function hessian_objective(x) end

"""
    weight_matrix(model) -> Union{Nothing, AbstractMatrix}

Return the weight matrix used in GMM estimation.

This is primarily used for inefficient GMM estimators or when implementing
the misspecified GMM variance form.

# Returns
- `AbstractMatrix`: m × m weight matrix
- `Nothing`: if not available (will default to optimal weight Ω⁻¹)

# Note
For efficient GMM, this is typically Ω⁻¹ where Ω is the covariance of moments.
For first-step or suboptimal GMM, this might be the identity or some other matrix.
The resulting matrix is then different.
"""
function weight_matrix(x) end

## TODO: Model checking should be done on whether the model is MLikeModel GMMLikeModel.

# StatsAPI.coef should already be implemented by most statistical models
# But we can provide a helpful error message if it's missing
function _check_coef(model)
    try
        StatsAPI.coef(model)
    catch MethodError
        t = typeof(model)
        error(
            "coef not implemented for type $t. " *
            "Please implement: StatsAPI.coef(::$(t)) -> AbstractVector",
        )
    end
end

# StatsAPI.nobs should be implemented by all statistical models
function _check_nobs(model)
    try
        StatsAPI.nobs(model)
    catch MethodError
        t = typeof(model)
        error(
            "nobs not implemented for type $t. " *
            "Please implement: StatsAPI.nobs(::$(t)) -> Integer",
        )
    end
end

"""
    _check_model_interface(model)

Verify that a model implements the minimum required interface.
"""
function _check_model_interface(model)
    # Check required methods
    try
        momentmatrix(model)
    catch MethodError
        t = typeof(model)
        error("Model type $t must implement CovarianceMatrices.momentmatrix")
    end

    _check_coef(model)
    _check_nobs(model)

    # Check that score is available when needed
    # (This will be checked in the specific variance form methods)
end

"""
    _check_dimensions(form::VarianceForm, model)

Check that model dimensions are compatible with the requested variance form.
Uses model type hierarchy when available, falls back to dimension checking.
"""

# Type-based checks for models using the hierarchy
function _check_dimensions(form::VarianceForm, model::MLikeModel)
    Z = momentmatrix(model)
    θ = StatsAPI.coef(model)
    m, k = size(Z, 2), length(θ)

    # For MLE models, we require m = k (exactly identified)
    if m != k
        throw(ArgumentError("MLikeModel requires exactly identified system: m=$m parameters but k=$k moment conditions"))
    end

    # cross_score is always available via default implementation
    # No additional checks needed for MLikeModel
end

function _check_dimensions(form::VarianceForm, model::GMMLikeModel)
    Z = momentmatrix(model)
    θ = StatsAPI.coef(model)
    m, k = size(Z, 2), length(θ)

    # For GMM models, we allow m >= k (at least identified)
    if m < k
        throw(ArgumentError("GMMLikeModel requires at least identified system: m=$m moment conditions but k=$k parameters"))
    end

    # cross_score is always available via default implementation
    # Only check hessian_objective for Misspecified form
    if form isa Misspecified
        hessian_result = hessian_objective(model)
        if hessian_result === nothing
            throw(ArgumentError("Misspecified form for GMM models requires hessian_objective(model) to return a k×k matrix where k=$k parameters"))
        end
    end
end

# Fallback dimension-based checks for models not using the type hierarchy
function _check_dimensions(form::VarianceForm, model)
    # For models that don't inherit from MLikeModel or GMMLikeModel,
    # we fall back to dimension checking
    Z = momentmatrix(model)
    θ = StatsAPI.coef(model)
    m, k = size(Z, 2), length(θ)

    # cross_score is always available via default implementation
    # Only check hessian_objective for GMM Misspecified form
    if m > k && form isa Misspecified
        if hessian_objective(model) === nothing
            throw(ArgumentError("Misspecified form for overidentified models (GMM-like) requires hessian_objective(model) to return a k×k matrix where k=$k parameters"))
        end
    end

    if m < k
        throw(ArgumentError("Invalid model: fewer moments (m=$m) than parameters (k=$k)"))
    end
end

"""
    _check_matrix_compatibility(form::VarianceForm, Z, cross_score, hessian_objective, W)

Check compatibility of provided matrices for manual API.
"""
function _check_matrix_compatibility(
        form::Information,
        Z::AbstractMatrix,
        cross_score_mat,
        hessian_objective,
        W
)
    n, m = size(Z)

    if hessian_objective !== nothing
        k_h, k_h2 = size(hessian_objective)
        if k_h != k_h2
            throw(
                ArgumentError(
                "hessian_objective must be square, got size $(size(hessian_objective))",
            ),
            )
        end
    end

    if cross_score_mat !== nothing
        m_cs, m_cs2 = size(cross_score_mat)
        if m_cs != m_cs2 || m_cs != m
            throw(
                ArgumentError(
                "cross_score must be m×m where m=$m moment conditions, got size $(size(cross_score_mat))",
            ),
            )
        end
    end

    if hessian_objective === nothing && cross_score_mat === nothing
        throw(ArgumentError("Information form requires either hessian_objective or cross_score"))
    end
end

function _check_matrix_compatibility(
        form::Misspecified,
        Z::AbstractMatrix,
        cross_score_mat,
        hessian_objective,
        W
)
    n, m = size(Z)

    if cross_score_mat === nothing
        throw(ArgumentError("Misspecified form requires cross_score matrix"))
    end

    m_cs, m_cs2 = size(cross_score_mat)
    if m_cs != m_cs2 || m_cs != m
        throw(
            ArgumentError(
            "cross_score must be m×m where m=$m moment conditions, got size $(size(cross_score_mat))",
        ),
        )
    end

    if W !== nothing
        w_m, w_m2 = size(W)
        if w_m != w_m2 || w_m != m
            throw(
                ArgumentError(
                "Weight matrix W must be m×m where m=$m, got size $(size(W))",
            ),
            )
        end
    end
end
