"""
Model integration interface for third-party estimators.

This module defines the duck-typed interface that third-party estimator
objects should implement to work with the CovarianceMatrices.jl API.
"""

"""
Type hierarchy
"""
abstract type MLikeModel <: StatsBase.StatisticalModel end
abstract type GMMLikeModel <: StatsBase.StatisticalModel end


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
    jacobian(model) -> AbstractMatrix

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
function jacobian(x)
    t = typeof(x)
    error("jacobian not implemented for type $t. " *
          "Please implement: CovarianceMatrices.jacobian(::$(t)) -> AbstractMatrix")
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
        error("coef not implemented for type $t. " *
              "Please implement: StatsBase.coef(::$(t)) -> AbstractVector")
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

    # Check that jacobian is available when needed
    # (This will be checked in the specific variance form methods)
end

"""
    _check_dimensions(form::VarianceForm, model)

Check that model dimensions are compatible with the requested variance form.
"""
function _check_dimensions(form::Information, model)
    Z = momentmatrix(model)
    θ = StatsBase.coef(model)
    m, k = size(Z, 2), length(θ)

    if m != k
        throw(ArgumentError("Information form requires exactly identified model (m = k), got m=$m, k=$k"))
    end
end

function _check_dimensions(form::Robust, model)
    Z = momentmatrix(model)
    θ = StatsBase.coef(model)
    m, k = size(Z, 2), length(θ)

    if m != k
        throw(ArgumentError("Robust form requires exactly identified model (m = k), got m=$m, k=$k"))
    end

    # Check that jacobian is available
    if jacobian(model) === nothing
        throw(ArgumentError("Robust form requires jacobian(model) to be implemented"))
    end
end

function _check_dimensions(form::CorrectlySpecified, model)
    Z = momentmatrix(model)
    θ = StatsBase.coef(model)
    m, k = size(Z, 2), length(θ)

    if m <= k
        throw(ArgumentError("CorrectlySpecified form requires overidentified model (m > k), got m=$m, k=$k"))
    end

    # Check that jacobian is available
    if jacobian(model) === nothing
        throw(ArgumentError("CorrectlySpecified form requires jacobian(model) to be implemented"))
    end
end

function _check_dimensions(form::Misspecified, model)
    Z = momentmatrix(model)
    θ = StatsBase.coef(model)
    m, k = size(Z, 2), length(θ)

    if m <= k
        throw(ArgumentError("Misspecified form requires overidentified model (m > k), got m=$m, k=$k"))
    end

    # Check that jacobian is available
    if jacobian(model) === nothing
        throw(ArgumentError("Misspecified form requires jacobian(model) to be implemented"))
    end
end

"""
    _check_matrix_compatibility(form::VarianceForm, Z, jacobian, objective_hessian, W)

Check compatibility of provided matrices for manual API.
"""
function _check_matrix_compatibility(form::Information, Z::AbstractMatrix,
    jacobian, objective_hessian, W)
    n, m = size(Z)

    if objective_hessian !== nothing
        k_h, k_h2 = size(objective_hessian)
        if k_h != k_h2
            throw(ArgumentError("objective_hessian must be square, got size $(size(objective_hessian))"))
        end
    end

    if jacobian !== nothing
        m_j, k_j = size(jacobian)
        if m_j != m
            throw(ArgumentError("jacobian first dimension ($m_j) must match moment matrix second dimension ($m)"))
        end
    end

    if objective_hessian === nothing && jacobian === nothing
        throw(ArgumentError("Information form requires either objective_hessian or jacobian"))
    end
end

function _check_matrix_compatibility(form::Union{Robust,CorrectlySpecified,Misspecified},
    Z::AbstractMatrix, jacobian, objective_hessian, W)
    n, m = size(Z)

    if jacobian === nothing
        throw(ArgumentError("$(typeof(form)) form requires jacobian matrix"))
    end

    m_j, k_j = size(jacobian)
    if m_j != m
        throw(ArgumentError("jacobian first dimension ($m_j) must match moment matrix second dimension ($m)"))
    end

    if form isa Union{CorrectlySpecified,Misspecified} && m <= k_j
        throw(ArgumentError("$(typeof(form)) form requires overidentified model (m > k), got m=$m, k=$k_j"))
    elseif form isa Robust && m != k_j
        throw(ArgumentError("Robust form requires exactly identified model (m = k), got m=$m, k=$k_j"))
    end

    if W !== nothing
        w_m, w_m2 = size(W)
        if w_m != w_m2 || w_m != m
            throw(ArgumentError("Weight matrix W must be m×m where m=$m, got size $(size(W))"))
        end
    end
end
