"""
Numerically stable variance computation methods.

This module implements the core variance computations using factorizations
instead of explicit matrix inversions for numerical stability.
"""

using LinearAlgebra

"""
    _compute_vcov(form::VarianceForm, model, Ω, W; kwargs...)

Dispatch to appropriate variance computation based on form.
"""
function _compute_vcov(
    form::Information,
    H,
    G,
    Ω,
    W;
    rcond_tol::Real = 1e-12,
    warn::Bool = true,
)
    # Check if we have a Hessian (MLE case) or need to use score (GMM case)
    if H !== nothing
        # MLE case: V = H⁻¹ (Fisher Information)
        Hinv, flag = ipinv(H)
        if warn && any(flag)
            @warn "The inverse hessian is not invertible. Correction to the smallest eigenvalues applied"
        end
        return Hinv
    else
        # GMM case: V = (G'Ω⁻¹G)⁻¹ (efficient GMM)
        Ωinv, flag = ipinv(Ω)
        if warn && any(flag)
            @warn "The inverse of Ω is not invertible. Correction to the smallest eigenvalues applied"
        end
        V, flag = ipinv(G' * Ωinv * G)
        if warn && any(flag)
            @warn "The variance matrix is not invertible. Correction to the smallest eigenvalue(s) applied"
        end
        return V
    end
end

# Misspecified form - dispatches based on model type context
# For MLE-like models (m=k): robust sandwich V = G⁻¹ΩG⁻ᵀ
# For GMM-like models (m>k): robust GMM V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹

function _compute_vcov(
    form::Misspecified,
    H,
    G,
    Ω,
    W;
    rcond_tol::Real = 1e-12,
    warn::Bool = true,
)
    m, k = size(G)

    if m == k
        # MLE-like: robust sandwich form V = G⁻¹ΩG⁻ᵀ
        Ginv, flag = ipinv(G)
        if warn && any(flag)
            @warn "The inverse of the score is not invertible. Correction to the smallest eigenvalues applied"
        end
        return Ginv' * Ω * Ginv
    else
        # GMM-like: robust GMM form V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹
        _compute_vcov_gmm_misspecified(H, G, Ω, W; rcond_tol = rcond_tol, warn = warn)
    end
end

function _compute_vcov_gmm_misspecified(
    H,
    G,
    Ω,
    W;
    rcond_tol::Real = 1e-12,
    warn::Bool = true,
)
    ## Ideally warn if correction to SVD was performed
    Ωinv, flag = ipinv(Ω)
    if warn && any(flag)
        @warn "The inverse of Ω is not invertible. Correction to the smallest eigenvalues applied"
    end
    B, flag = ipinv(G' * Ωinv * G)
    if warn && any(flag)
        @warn "The inverse of G'ΩG is not invertible. Correction to the smallest eigenvalues applied"
    end
    Hinv, flag = ipinv(H)
    if warn && any(flag)
        @warn "The inverse of H is not invertible. Correction to the smallest eigenvalues applied"
    end
    return Hinv' * B * Hinv
end


function ipinv(
    A::AbstractMatrix{T};
    atol::Real = 0.0,
    rtol::Real = (eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol),
) where {T}
    m, n = size(A)
    Tout = typeof(zero(T) / sqrt(oneunit(T) + oneunit(T)))
    if m == 0 || n == 0
        return similar(A, Tout, (n, m))
    end
    if isdiag(A)
        indA = LinearAlgebra.diagind(A)
        dA = view(A, indA)
        maxabsA = maximum(abs, dA)
        tol = max(rtol * maxabsA, atol)
        B = fill!(similar(A, Tout, (n, m)), 0)
        indB = LinearAlgebra.diagind(B)
        B[indB] .= (x -> abs(x) > tol ? pinv(x) : zero(x)).(dA)
        return B, .!(abs.(dA) .> tol)
    end
    SVD = svd(A)
    tol2 = max(rtol * maximum(SVD.S), atol)
    Stype = eltype(SVD.S)
    Sinv = fill!(similar(A, Stype, length(SVD.S)), 0)
    index = SVD.S .> tol2
    Sinv[index] .= pinv.(view(SVD.S, index))
    return SVD.Vt' * (Diagonal(Sinv) * SVD.U'), .!index
end

function ipinv(x::Number)
    xi = inv(x)
    return ifelse(isfinite(xi), xi, zero(xi))
end
