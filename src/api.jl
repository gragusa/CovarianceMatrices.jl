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

Compute variance-covariance matrix for a model using specified estimator and form.

# Arguments
- `ve::AVarEstimator`: Variance estimator (HAC, HC, CR, etc.)
- `form::VarianceForm`: Variance form (Information, Robust, CorrectlySpecified, Misspecified)
- `model`: Statistical model implementing required interface methods

# Keyword Arguments
- `W::Union{Nothing,AbstractMatrix}=nothing`: Optional weight matrix for GMM misspecified
- `scale::Symbol=:n`: Scaling for Ω (:n for 1/n scaling)
- `rcond_tol::Real=1e-12`: Tolerance for rank condition in pseudo-inverse
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Variance-covariance matrix
"""
function StatsBase.vcov(ve::AVarEstimator, form::VarianceForm, model;
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
    Z = CovarianceMatrices.momentmatrix(model)
    n = nobs(model)

    # Compute long-run covariance
    Ω = aVar(ve, Z)
    G = score(model)
    H = objective_hessian(model)
    # Dispatch to appropriate computation
    V = _compute_vcov(form, H, G, Ω, W; rcond_tol=rcond_tol, warn=warn)

    return Symmetric(rdiv!(V, n))
end

"""
    vcov(ve::AVarEstimator, form::VarianceForm, Z::AbstractMatrix; kwargs...)

Manual variance computation from moment matrix.

# Arguments
- `ve::AVarEstimator`: Variance estimator
- `form::VarianceForm`: Variance form
- `Z::AbstractMatrix`: Moment matrix (n × m)

# Keyword Arguments
- `score::Union{Nothing,AbstractMatrix}=nothing`: Jacobian matrix G (m × k)
- `objective_hessian::Union{Nothing,AbstractMatrix}=nothing`: Hessian matrix H (k × k)
- `W::Union{Nothing,AbstractMatrix}=nothing`: Weight matrix (m × m)
- `rcond_tol::Real=1e-12`: Tolerance for rank condition
"""
function StatsBase.vcov(ve::AVarEstimator, form::VarianceForm, Z::AbstractMatrix;
    score::Union{Nothing,AbstractMatrix}=nothing,
    objective_hessian::Union{Nothing,AbstractMatrix}=nothing,
    W::Union{Nothing,AbstractMatrix}=nothing,
    rcond_tol::Real=1e-12)


    n, m = size(Z)

    # Check what's available and what's required
    _check_matrix_compatibility(form, Z, score, objective_hessian, W)

    # Compute long-run covariance
    Ω = aVar(ve, Z; scale=false)

    # Compute variance
    H = objective_hessian
    G = score
    V = _compute_vcov(form, H, G, Ω, W; rcond_tol=rcond_tol, warn=false)

    return Symmetric(rdiv!(V, n))
end


"""
    stderror(ve::AVarEstimator, args...; kwargs...)

Compute standard errors from variance-covariance matrix.
"""
function StatsBase.stderror(ve::AVarEstimator, args...; kwargs...)
    V = StatsBase.vcov(ve, args...; kwargs...)
    return sqrt.(diag(V))
end
