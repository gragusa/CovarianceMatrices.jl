"""
Model integration interface for third-party estimators.

This module defines the duck-typed interface that third-party estimator
objects should implement to work with the CovarianceMatrices.jl API.
"""

"""
Type hierarchy for statistical models
"""
abstract type MLikeModel <: StatsBase.StatisticalModel end
abstract type GMMLikeModel <: StatsBase.StatisticalModel end

# Forward declarations for variance forms (defined in variance_forms.jl)
abstract type VarianceForm end
struct Information <: VarianceForm end
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
    score(model) -> AbstractMatrix

Return the Jacobian matrix of the moment conditions.

This is the matrix of partial derivatives ∂ḡ/∂θ' evaluated at the estimated
parameters, where ḡ is the sample mean of the moment conditions.

# Returns
- `AbstractMatrix`: m × k matrix where m is the number of moment conditions
  and k is the number of parameters.

# Note
For exactly identified models (m = k), this equals the negative inverse
of the Hessian of the objective function in many cases.
"""
function score(x)
    t = typeof(x)
    error(
        "score not implemented for type $t. " *
        "Please implement: CovarianceMatrices.score(::$(t)) -> AbstractMatrix",
    )
end

"""
    objective_hessian(model) -> Union{Nothing, AbstractMatrix}

Return the Hessian matrix of the estimator's objective function.

This is the matrix of second derivatives of the objective function with
respect to the parameters, evaluated at the estimated parameters.

# Returns
- `AbstractMatrix`: k × k matrix where k is the number of parameters
- `Nothing`: if not available or not applicable

# Note
For MLE, this is the Hessian of the negative log-likelihood.
For exactly identified models, this often equals the Jacobian matrix.
"""
function objective_hessian(x)
    # Default: not available
    return nothing
end

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
function weight_matrix(x)
    # Default: not available (will use optimal weight)
    return nothing
end

## TODO: Model checking should be done on whether the model is MLikeModel GMMLikeModel.

# StatsBase.coef should already be implemented by most statistical models
# But we can provide a helpful error message if it's missing
function _check_coef(model)
    try
        StatsBase.coef(model)
    catch MethodError
        t = typeof(model)
        error(
            "coef not implemented for type $t. " *
            "Please implement: StatsBase.coef(::$(t)) -> AbstractVector",
        )
    end
end

# StatsBase.nobs should be implemented by all statistical models
function _check_nobs(model)
    try
        StatsBase.nobs(model)
    catch MethodError
        t = typeof(model)
        error(
            "nobs not implemented for type $t. " *
            "Please implement: StatsBase.nobs(::$(t)) -> Integer",
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
# Both forms work with both model types now, so just allow them
_check_dimensions(form::VarianceForm, model::MLikeModel) = nothing
_check_dimensions(form::VarianceForm, model::GMMLikeModel) = nothing

# Fallback dimension-based checks for models not using the type hierarchy
function _check_dimensions(form::VarianceForm, model)
    # For models that don't inherit from MLikeModel or GMMLikeModel,
    # we fall back to dimension checking
    Z = momentmatrix(model)
    θ = StatsBase.coef(model)
    m, k = size(Z, 2), length(θ)

    # For exactly identified models (m = k), assume MLE-like
    if m == k
        # Check that required methods are available for Misspecified form
        if form isa Misspecified && score(model) === nothing
            throw(
                ArgumentError("Misspecified form requires score(model) to be implemented"),
            )
        end
        # For overidentified models (m > k), assume GMM-like
    elseif m > k
        # Both forms require score for GMM
        if score(model) === nothing
            throw(
                ArgumentError(
                "$(typeof(form)) form requires score(model) to be implemented for overidentified models",
            ),
            )
        end
    else
        throw(ArgumentError("Invalid model: fewer moments (m=$m) than parameters (k=$k)"))
    end
end

"""
    _check_matrix_compatibility(form::VarianceForm, Z, score, objective_hessian, W)

Check compatibility of provided matrices for manual API.
"""
function _check_matrix_compatibility(
        form::Information,
        Z::AbstractMatrix,
        score,
        objective_hessian,
        W
)
    n, m = size(Z)

    if objective_hessian !== nothing
        k_h, k_h2 = size(objective_hessian)
        if k_h != k_h2
            throw(
                ArgumentError(
                "objective_hessian must be square, got size $(size(objective_hessian))",
            ),
            )
        end
    end

    if score !== nothing
        m_j, k_j = size(score)
        if m_j != m
            throw(
                ArgumentError(
                "score first dimension ($m_j) must match moment matrix second dimension ($m)",
            ),
            )
        end
    end

    if objective_hessian === nothing && score === nothing
        throw(ArgumentError("Information form requires either objective_hessian or score"))
    end
end

function _check_matrix_compatibility(
        form::Misspecified,
        Z::AbstractMatrix,
        score,
        objective_hessian,
        W
)
    n, m = size(Z)

    if score === nothing
        throw(ArgumentError("Misspecified form requires score matrix"))
    end

    m_j, k_j = size(score)
    if m_j != m
        throw(
            ArgumentError(
            "score first dimension ($m_j) must match moment matrix second dimension ($m)",
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
