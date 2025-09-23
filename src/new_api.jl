"""
New unified API for covariance matrix estimation.

This module implements the improved API design that provides a single interface
for third-party estimators to obtain asymptotic covariance matrices under
either correct specification or misspecification.
"""

using LinearAlgebra
using StatsBase

"""
    vcov(ve::AVarEstimator, form::VarianceForm, model; kwargs...)
    vcov(ve::AVarEstimator, model; form::Symbol=:auto, kwargs...)

Compute variance-covariance matrix for a model using specified estimator and form.

# Arguments
- `ve::AVarEstimator`: Variance estimator (HAC, HC, CR, etc.)
- `form::VarianceForm`: Variance form or Symbol (:auto, :information, :robust, etc.)
- `model`: Statistical model implementing required interface methods

# Keyword Arguments
- `W::Union{Nothing,AbstractMatrix}=nothing`: Optional weight matrix for GMM misspecified
- `scale::Symbol=:n`: Scaling for Ω (:n for 1/n scaling)
- `rcond_tol::Real=1e-12`: Tolerance for rank condition in pseudo-inverse
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Variance-covariance matrix

# Form Resolution
When `form=:auto`:
- If m == k (M-like): defaults to `:robust` (safe)
- If m > k (GMM-like): defaults to `:correctly_specified` (efficient)
"""
function vcov_new(ve::AVarEstimator, form::VarianceForm, model;
                       W::Union{Nothing,AbstractMatrix}=nothing,
                       scale::Symbol=:n,
                       rcond_tol::Real=1e-12,
                       check::Bool=true,
                       warn::Bool=true)

    if check
        _check_model_interface(model)
        _check_dimensions(form, model)
    end

    # Get required matrices
    Z = momentmatrix(model)
    n = size(Z, 1)

    # Compute long-run covariance
    Ω = aVar(ve, Z; scale=false)  # Don't scale here - we'll handle it

    # Dispatch to appropriate computation
    V = _compute_vcov(form, model, Ω, W; rcond_tol=rcond_tol, warn=warn)

    # Apply scaling
    _scale_vcov!(V, scale, n)

    return Symmetric(V)
end

# Symbol-based interface for backward compatibility
function vcov_new(ve::AVarEstimator, model;
                       form::Symbol=:auto,
                       W::Union{Nothing,AbstractMatrix}=nothing,
                       scale::Symbol=:n,
                       rcond_tol::Real=1e-12,
                       check::Bool=true,
                       warn::Bool=true)

    if form == :auto
        variance_form = auto_form(model)
    else
        variance_form = symbol_to_form(form)
    end

    return vcov_new(ve, variance_form, model; W=W, scale=scale, rcond_tol=rcond_tol,
                    check=check, warn=warn)
end

"""
    vcov(ve::AVarEstimator, form::VarianceForm, Z::AbstractMatrix; kwargs...)

Manual variance computation from moment matrix.

# Arguments
- `ve::AVarEstimator`: Variance estimator
- `form::VarianceForm`: Variance form
- `Z::AbstractMatrix`: Moment matrix (n × m)

# Keyword Arguments
- `jacobian::Union{Nothing,AbstractMatrix}=nothing`: Jacobian matrix G (m × k)
- `objective_hessian::Union{Nothing,AbstractMatrix}=nothing`: Hessian matrix H (k × k)
- `W::Union{Nothing,AbstractMatrix}=nothing`: Weight matrix (m × m)
- `rcond_tol::Real=1e-12`: Tolerance for rank condition
"""
function vcov_new(ve::AVarEstimator, form::VarianceForm, Z::AbstractMatrix;
                       jacobian::Union{Nothing,AbstractMatrix}=nothing,
                       objective_hessian::Union{Nothing,AbstractMatrix}=nothing,
                       W::Union{Nothing,AbstractMatrix}=nothing,
                       rcond_tol::Real=1e-12)

    n, m = size(Z)

    # Check what's available and what's required
    _check_matrix_compatibility(form, Z, jacobian, objective_hessian, W)

    # Compute long-run covariance
    Ω = aVar(ve, Z; scale=false)

    # Create a mock model object for dispatch
    mock_model = MatrixModel(Z, jacobian, objective_hessian, W)

    # Compute variance
    V = _compute_vcov(form, mock_model, Ω, W; rcond_tol=rcond_tol, warn=false)

    return Symmetric(V ./ n)
end

"""
    stderror(ve::AVarEstimator, args...; kwargs...)

Compute standard errors from variance-covariance matrix.
"""
function stderror_new(ve::AVarEstimator, args...; kwargs...)
    V = vcov_new(ve, args...; kwargs...)
    return sqrt.(diag(V))
end

# Internal helper struct for matrix-based API
struct MatrixModel{T<:Real}
    Z::AbstractMatrix{T}
    jacobian::Union{Nothing,AbstractMatrix{T}}
    objective_hessian::Union{Nothing,AbstractMatrix{T}}
    W::Union{Nothing,AbstractMatrix{T}}
end

# Implement required interface for MatrixModel
momentmatrix(m::MatrixModel) = m.Z
jacobian(m::MatrixModel) = m.jacobian
objective_hessian(m::MatrixModel) = m.objective_hessian
weight_matrix(m::MatrixModel) = m.W
StatsBase.coef(m::MatrixModel) = zeros(eltype(m.Z), _infer_k(m))

function _infer_k(m::MatrixModel)
    if m.jacobian !== nothing
        return size(m.jacobian, 2)
    elseif m.objective_hessian !== nothing
        return size(m.objective_hessian, 1)
    else
        # For exactly identified case, k = m
        return size(m.Z, 2)
    end
end