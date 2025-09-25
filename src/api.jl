
"""
New unified API for covariance matrix estimation.

This module implements the improved API design that provides a single interface
for third-party estimators to obtain asymptotic covariance matrices under
either correct specification or misspecification.
"""

using LinearAlgebra
using StatsBase

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::VarianceForm, model; kwargs...)

Compute variance-covariance matrix for a model using specified estimator and form.

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator (HAC, HC, CR, etc.)
- `form::VarianceForm`: Variance form (Information, Robust, CorrectlySpecified, Misspecified)
- `model`: Statistical model implementing required interface methods

# Keyword Arguments
- `W::Union{Nothing,AbstractMatrix}=nothing`: Optional weight matrix for GMM misspecified
- `scale::Symbol=:n`: Scaling for Ω (:n for 1/n scaling)
- `cond_atol::Union{Nothing,Real}=nothing`: Absolute tolerance for pseudo-inverse (default: 0.0)
- `cond_rtol::Union{Nothing,Real}=nothing`: Relative tolerance for pseudo-inverse (default: machine eps × min(size))
- `debug::Bool=false`: Print detailed debug information about matrix inversions
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues (automatically true when debug=true)

# Returns
- `Matrix{Float64}`: Variance-covariance matrix
"""
function StatsBase.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::VarianceForm,
        model;
        W::Union{Nothing, AbstractMatrix} = nothing,
        scale::Symbol = :n,
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        check::Bool = true,
        warn::Bool = true
)
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
    V = _compute_vcov(form, H, G, Ω, W; cond_atol = cond_atol,
        cond_rtol = cond_rtol, debug = debug, warn = warn)

    return Symmetric(rdiv!(V, n))
end

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::VarianceForm, Z::AbstractMatrix; kwargs...)

Manual variance computation from moment matrix.

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator
- `form::VarianceForm`: Variance form
- `Z::AbstractMatrix`: Moment matrix (n × m)

# Keyword Arguments
- `score::Union{Nothing,AbstractMatrix}=nothing`: Jacobian matrix G (m × k)
- `objective_hessian::Union{Nothing,AbstractMatrix}=nothing`: Hessian matrix H (k × k)
- `W::Union{Nothing,AbstractMatrix}=nothing`: Weight matrix (m × m)
- `rcond_tol::Real=1e-12`: Tolerance for rank condition
"""

function StatsBase.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::VarianceForm,
        Z::AbstractMatrix;
        score::Union{Nothing, AbstractMatrix} = nothing,
        objective_hessian::Union{Nothing, AbstractMatrix} = nothing,
        weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false
)
    n, m = size(Z)

    # Check what's available and what's required
    _check_matrix_compatibility(form, Z, score, objective_hessian, weight_matrix)

    # Compute long-run covariance
    Ω = aVar(ve, Z)

    # Compute variance
    H = objective_hessian
    G = score
    V = _compute_vcov(form, H, G, Ω, weight_matrix; cond_atol = cond_atol,
        cond_rtol = cond_rtol, debug = debug, warn = false)

    return Symmetric(rdiv!(V, n))
end

"""
    stderror(ve::AbstractAsymptoticVarianceEstimator, args...; kwargs...)

Compute standard errors from variance-covariance matrix.
"""
function StatsBase.stderror(ve::AbstractAsymptoticVarianceEstimator, args...; kwargs...)
    V = StatsBase.vcov(ve, args...; kwargs...)
    return sqrt.(diag(V))
end

# ============================================================================
# VARHAC-specific implementations
# ============================================================================

"""
    vcov(ve::VARHAC, form::VarianceForm, model; kwargs...)

VARHAC-specific vcov implementation. VARHAC estimates directly provide the
spectral density at frequency zero, which is the desired variance-covariance
matrix for parameter estimates.

For VARHAC:
- Information form: Return S(0) directly since VARHAC estimates the spectral density
- Misspecified form: Return S(0) directly (same as Information for VARHAC)

VARHAC is inherently robust to serial correlation and provides consistent
estimates under both correct specification and misspecification.
"""
function StatsBase.vcov(
        ve::VARHAC,
        form::VarianceForm,
        model;
        scale::Bool = true,
        kwargs...
)
    # Extract moment matrix from model
    Z = momentmatrix(model)
    return vcov(ve, form, Z; kwargs...)
end

function StatsBase.vcov(
        ve::VARHAC,
        form::VarianceForm,
        Z::AbstractMatrix;
        scale::Bool = true,
        kwargs...
)
    # For VARHAC, both Information and Misspecified forms give the same result
    # since VARHAC directly estimates the spectral density at frequency zero
    S_hat = aVar(ve, Z; scale = scale, kwargs...)
    return Symmetric(S_hat)
end
