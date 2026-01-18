
"""
New unified API for covariance matrix estimation.

This module implements the improved API design that provides a single interface
for third-party estimators to obtain asymptotic covariance matrices under
either correct specification or misspecification.
"""

# ============================================================================
# MLikeModel Methods
# ============================================================================

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::Information, model::MLikeModel; kwargs...)

Compute variance-covariance matrix for MLE-like models under correct specification assumption.

For MLikeModel + Information:
- If hessian_objective available: V = inv(H) where H is the Hessian
- Otherwise: V = inv(G) where G is the cross_score matrix

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator (HAC, HC, etc.)
- `form::Information`: Information form (assumes correct specification)
- `model::MLikeModel`: Maximum likelihood model

# Keyword Arguments
- `cond_atol::Union{Nothing,Real}=nothing`: Absolute tolerance for pseudo-inverse
- `cond_rtol::Union{Nothing,Real}=nothing`: Relative tolerance for pseudo-inverse
- `debug::Bool=false`: Print detailed debug information
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Variance-covariance matrix (Fisher Information inverse)
"""
function StatsAPI.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::Information,
        model::MLikeModel;
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

    # Get Hessian if available, otherwise use cross_score
    H = hessian_objective(model)

    if H !== nothing
        # Use Hessian: V = inv(H)
        V = _compute_mle_information(H; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    else
        # Fall back to cross_score: V = inv(G)
        G = cross_score(model)
        V = _compute_mle_information(G; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    end

    return V
end

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::Misspecified, model::MLikeModel; kwargs...)

Compute robust sandwich variance for MLE-like models allowing misspecification.

For MLikeModel + Misspecified: V = inv(H) * G * inv(H)
Both hessian_objective (H) and cross_score (G) are required.

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator
- `form::Misspecified`: Misspecified form (robust to misspecification)
- `model::MLikeModel`: Maximum likelihood model

# Keyword Arguments
- `cond_atol::Union{Nothing,Real}=nothing`: Absolute tolerance for pseudo-inverse
- `cond_rtol::Union{Nothing,Real}=nothing`: Relative tolerance for pseudo-inverse
- `debug::Bool=false`: Print detailed debug information
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Robust sandwich variance-covariance matrix
"""
function StatsAPI.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::Misspecified,
        model::MLikeModel;
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
    Z = momentmatrix(model)
    Ω = aVar(ve, Z; scale = false)
    H = hessian_objective(model)

    # Both H and G are required for misspecified MLE
    if H === nothing
        throw(ArgumentError("Misspecified form for MLikeModel requires hessian_objective to be implemented"))
    end

    # Compute V = inv(H) * G * inv(H)
    V = _compute_mle_misspecified(H, Ω; cond_atol = cond_atol,
        cond_rtol = cond_rtol, debug = debug, warn = warn)

    return V
end

# ============================================================================
# GMMLikeModel Methods
# ============================================================================

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::Information, model::GMMLikeModel; kwargs...)

Compute variance-covariance matrix for GMM-like models under correct specification.

For GMMLikeModel + Information: V = inv(G' * inv(Ω) * G)
This is the efficient GMM variance formula.

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator
- `form::Information`: Information form (assumes correct specification)
- `model::GMMLikeModel`: GMM model

# Keyword Arguments
- `cond_atol::Union{Nothing,Real}=nothing`: Absolute tolerance for pseudo-inverse
- `cond_rtol::Union{Nothing,Real}=nothing`: Relative tolerance for pseudo-inverse
- `debug::Bool=false`: Print detailed debug information
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Efficient GMM variance-covariance matrix
"""
function StatsAPI.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::Information,
        model::GMMLikeModel;
        W::Union{Nothing, AbstractMatrix} = nothing,
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
    Z = momentmatrix(model)
    Ω = aVar(ve, Z; scale = false)
    G = jacobian_momentfunction(model)

    # Get weight matrix (use identity if not provided)
    if W === nothing
        # Optimal GMM: W = inv(Ω)
        # Formula: V = inv(G' * inv(Ω) * G)
        V = _compute_gmm_information(G, Ω; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    else
        # Suboptimal GMM with provided weight matrix
        # Formula: V = inv(G' * W * inv(Ω) * W * G)
        V = _compute_gmm_information_weighted(G, Ω, W; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    end

    return V
end

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, form::Misspecified, model::GMMLikeModel; kwargs...)

Compute robust GMM variance allowing misspecification.

For GMMLikeModel + Misspecified: V = inv(H) * [inv(G' * inv(Ω) * G)] * inv(H)
Requires both hessian_objective (H) and cross_score (G).

# Arguments
- `ve::AbstractAsymptoticVarianceEstimator`: Variance estimator
- `form::Misspecified`: Misspecified form (robust to misspecification)
- `model::GMMLikeModel`: GMM model

# Keyword Arguments
- `W::Union{Nothing,AbstractMatrix}=nothing`: Optional weight matrix
- `cond_atol::Union{Nothing,Real}=nothing`: Absolute tolerance for pseudo-inverse
- `cond_rtol::Union{Nothing,Real}=nothing`: Relative tolerance for pseudo-inverse
- `debug::Bool=false`: Print detailed debug information
- `check::Bool=true`: Perform dimension and compatibility checks
- `warn::Bool=true`: Issue warnings for potential issues

# Returns
- `Matrix{Float64}`: Robust GMM variance-covariance matrix
"""
function StatsAPI.vcov(
        ve::AbstractAsymptoticVarianceEstimator,
        form::Misspecified,
        model::GMMLikeModel;
        W::Union{Nothing, AbstractMatrix} = nothing,
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
    Z = momentmatrix(model)
    Ω = aVar(ve, Z; scale = false)
    G = jacobian_momentfunction(model)
    H = hessian_objective(model)

    # hessian_objective is required for GMM Misspecified
    if H === nothing
        throw(ArgumentError("Misspecified form for GMMLikeModel requires hessian_objective to be implemented"))
    end

    # Compute robust GMM variance
    if W === nothing
        # Formula: V = inv(H) * inv(G' * inv(Ω) * G) * inv(H)
        V = _compute_gmm_misspecified(H, G, Ω, nothing; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    else
        # Formula: V = inv(H) * inv(G' * W * inv(Ω) * W * G) * inv(H)
        V = _compute_gmm_misspecified(H, G, Ω, W; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    end

    return V
end

"""
    stderror(ve::AbstractAsymptoticVarianceEstimator, args...; kwargs...)

Compute standard errors from variance-covariance matrix.
"""
function StatsAPI.stderror(ve::AbstractAsymptoticVarianceEstimator, args...; kwargs...)
    V = StatsAPI.vcov(ve, args...; kwargs...)
    return sqrt.(diag(V))
end

# # ============================================================================
# # VARHAC-specific implementations
# # ============================================================================

# """
#     vcov(ve::VARHAC, form::VarianceForm, model; kwargs...)

# VARHAC-specific vcov implementation. VARHAC estimates directly provide the
# spectral density at frequency zero, which is the desired variance-covariance
# matrix for parameter estimates.

# For VARHAC:
# - Information form: Return S(0) directly since VARHAC estimates the spectral density
# - Misspecified form: Return S(0) directly (same as Information for VARHAC)

# VARHAC is inherently robust to serial correlation and provides consistent
# estimates under both correct specification and misspecification.
# """
# function StatsAPI.vcov(
#         ve::VARHAC,
#         form::VarianceForm,
#         model;
#         scale::Bool = true,
#         kwargs...
# )
#     # Extract moment matrix from model
#     Z = momentmatrix(model)
#     return vcov(ve, form, Z; kwargs...)
# end

# function StatsAPI.vcov(
#         ve::VARHAC,
#         form::VarianceForm,
#         Z::AbstractMatrix;
#         scale::Bool = true,
#         kwargs...
# )
#     # For VARHAC, both Information and Misspecified forms give the same result
#     # since VARHAC directly estimates the spectral density at frequency zero
#     S_hat = aVar(ve, Z; scale = scale, kwargs...)
#     return S_hat
# end
