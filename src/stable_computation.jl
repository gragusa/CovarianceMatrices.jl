"""
Numerically stable variance computation methods.

This module implements the core variance computations using factorizations
instead of explicit matrix inversions for numerical stability.
"""

using LinearAlgebra

# ============================================================================
# MLikeModel Computations
# ============================================================================

"""
    _compute_mle_information(M; kwargs...)

Compute V = inv(M) for MLE Information form.
M can be either H (hessian) or G (cross_score).

For unscaled inputs (H and G are sums, not averages):
V = inv(M)
"""
function _compute_mle_information(
        M::AbstractMatrix;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(M)))) * min(size(M)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # V = inv(M)
    Minv, flag, svals = ipinv(M; atol = atol, rtol = rtol_val)
    _debug_report_inversion("Fisher Information Matrix", flag, svals, "$(size(M))", debug,
        warn)

    return Minv
end

"""
    _compute_mle_misspecified(H, Ω; kwargs...)

Compute robust sandwich variance for MLE: V = inv(H) * Ω * inv(H)

For unscaled inputs:
- H: Hessian (sum, not average)
- Ω: Long-run covariance of scores (sum, not average)

The formula is: V = inv(H) * Ω * inv(H)
"""
function _compute_mle_misspecified(
        H::AbstractMatrix,
        Ω::AbstractMatrix;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # Invert H
    Hinv, flag_h, svals_h = ipinv(H; atol = atol, rtol = rtol_val)
    _debug_report_inversion("H (Hessian)", flag_h, svals_h, "$(size(H))", debug, warn)

    # Compute V = Hinv' * Ω * Hinv
    return Hinv' * Ω * Hinv
end

# ============================================================================
# GMMLikeModel Computations
# ============================================================================

"""
    _compute_gmm_information(G, Ω; kwargs...)

Compute efficient GMM variance: V = inv(G' * inv(Ω) * G)

For unscaled inputs:
- G: Jacobian matrix (sum, not average)
- Ω: Long-run covariance (sum, not average)

The formula is: V = inv(G' * inv(Ω) * G)
"""
function _compute_gmm_information(
        G::AbstractMatrix,
        Ω::AbstractMatrix;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # Invert Ω
    Ωinv, flag_omega, svals_omega = ipinv(Ω; atol = atol, rtol = rtol_val)
    _debug_report_inversion("Omega", flag_omega, svals_omega, "$(size(Ω))", debug, warn)

    # Compute G' * Ωinv * G
    G_omega_G = G' * Ωinv * G

    # Invert to get variance
    V, flag_v, svals_v = ipinv(G_omega_G; atol = atol, rtol = rtol_val)
    _debug_report_inversion("G'Omega^(-1)G", flag_v, svals_v, "$(size(G_omega_G))", debug,
        warn)

    return V
end

"""
    _compute_gmm_information_weighted(G, Ω, W; kwargs...)

Compute GMM variance with suboptimal weight matrix: V = inv(G'WG) * G'WΩWG * inv(G'WG)

This is the standard sandwich formula for GMM with arbitrary weight matrix W.
When W ≠ inv(Ω), the variance is not simply inv(G'WG) — the full sandwich is needed
even under correct specification.

For unscaled inputs:
- G: Jacobian matrix (sum, not average)
- Ω: Long-run covariance (sum, not average)
- W: Weight matrix (m × m)

The formula is: V = inv(G'WG) * (G'WΩWG) * inv(G'WG)
"""
function _compute_gmm_information_weighted(
        G::AbstractMatrix,
        Ω::AbstractMatrix,
        W::AbstractMatrix;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # Compute and invert G'WG (the "bread")
    GWG = G' * W * G
    GWGinv, flag_gwg, svals_gwg = ipinv(GWG; atol = atol, rtol = rtol_val)
    _debug_report_inversion("G'WG", flag_gwg, svals_gwg, "$(size(GWG))", debug, warn)

    # Compute the "meat": G'WΩWG
    meat = G' * (W * Ω * W) * G

    # Sandwich: V = inv(G'WG) * G'WΩWG * inv(G'WG)
    return GWGinv' * meat * GWGinv
end

"""
    _compute_gmm_misspecified(H, G, Ω, W; kwargs...)

Compute robust GMM variance:
- If W is nothing: V = inv(H) * (G' * inv(Ω) * G) * inv(H) (efficient GMM)
- If W is provided: V = inv(H) * (G' * W * Ω * W * G) * inv(H) (suboptimal GMM)

For unscaled inputs:
- H: Hessian (sum, not average)
- G: Jacobian matrix (sum, not average)
- Ω: Long-run covariance (sum, not average)
- W: Optional weight matrix (if nothing, uses efficient weight inv(Ω))
"""
function _compute_gmm_misspecified(
        H::AbstractMatrix,
        G::AbstractMatrix,
        Ω::AbstractMatrix,
        W::Union{Nothing, AbstractMatrix};
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # Compute the "meat" B of the sandwich
    if W === nothing
        # Efficient GMM (W = inv(Ω)): B = G'W Ω W G = G' inv(Ω) G
        Ωinv, flag_omega, svals_omega = ipinv(Ω; atol = atol, rtol = rtol_val)
        _debug_report_inversion("Ω", flag_omega, svals_omega, "$(size(Ω))", debug, warn)
        B = G' * Ωinv * G
    else
        # Suboptimal weight: B = G' * W * Ω * W * G
        B = G' * (W * Ω * W) * G
    end

    # Invert H
    Hinv, flag_h, svals_h = ipinv(H; atol = atol, rtol = rtol_val)
    _debug_report_inversion("H (Hessian)", flag_h, svals_h, "$(size(H))", debug, warn)

    # Compute V = inv(H) * B * inv(H)
    return Hinv' * B * Hinv
end

function _debug_report_inversion(matrix_name::String, flag::AbstractVector{Bool},
        svals::Vector, size_info::String, debug::Bool, warn::Bool)
    if any(flag) && !isempty(svals)
        n_problematic = sum(flag)
        min_sval = minimum(svals)
        max_sval = maximum(svals)
        problematic_svals = svals[findall(flag)]  # Use findall for proper indexing

        if debug
            println("🔍 DEBUG: Matrix inversion issue detected")
            println("   Matrix: $matrix_name ($size_info)")
            println("   Problematic singular values: $n_problematic/$(length(svals))")
            println("   Singular value range: [$(min_sval), $(max_sval)]")
            if !isempty(problematic_svals)
                println("   Problematic values: $(sort(problematic_svals))")
            end
            println("   Condition number: $(max_sval / max(min_sval, eps()))")
            println()
        elseif warn
            @warn "The inverse of $matrix_name is not invertible. Correction to the smallest eigenvalues applied ($(n_problematic)/$(length(svals)) values affected)"
        end
    end
end

function ipinv(
        A::AbstractMatrix{T};
        atol::Real = 0.0,
        rtol::Real = (eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol)
) where {T}
    m, n = size(A)
    Tout = typeof(zero(T) / sqrt(oneunit(T) + oneunit(T)))
    if m == 0 || n == 0
        return similar(A, Tout, (n, m)), Bool[], T[]
    end
    if isdiag(A)
        indA = LinearAlgebra.diagind(A)
        dA = view(A, indA)
        maxabsA = maximum(abs, dA)
        tol = max(rtol * maxabsA, atol)
        B = fill!(similar(A, Tout, (n, m)), 0)
        indB = LinearAlgebra.diagind(B)
        B[indB] .= (x -> abs(x) > tol ? pinv(x) : zero(x)).(dA)
        problematic = .!(abs.(dA) .> tol)
        return B, problematic, abs.(dA)  # Return diagonal values as "singular values"
    end
    SVD = svd(A)
    tol2 = max(rtol * maximum(SVD.S), atol)
    Stype = eltype(SVD.S)
    Sinv = fill!(similar(A, Stype, length(SVD.S)), 0)
    index = SVD.S .> tol2
    Sinv[index] .= pinv.(view(SVD.S, index))
    problematic = .!index
    return SVD.Vt' * (Diagonal(Sinv) * SVD.U'), problematic, SVD.S
end

function ipinv(x::Number)
    xi = inv(x)
    return ifelse(isfinite(xi), xi, zero(xi))
end
