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
function _compute_vcov(form::Information, H, G, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    ## Ideally warn if correction to SVD was performed
    Hinv, flag = ipinv(H)
    if warn && any(flag)
        @warn "The inverse hessian is not invertible. Correction to the smallest eigenvalues applied"
    end
    return Hinv
end

function _compute_vcov(form::Robust, H, G, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    ## Ideally warn if correction to SVD was performed
    Ginv, flag = ipinv(G)
    if warn && any(flag)
        @warn "The inverse of the jacobian is not invertible. Correction to the smallest eigenvalues applied"
    end
    return Ginv' * Ω * Ginv
end

function _compute_vcov(form::CorrectlySpecified, H, G, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    ## Ideally warn if correction to SVD was performed
    Ωinv, flag = ipinv(Ω)
    if warn && any(flag)
        @warn "The inverse of Ω is not invertible. Correction to the smallest eigenvalues applied"
    end
    V, flag = ipinv(G' * Ωinv * G)
    if warn && any(flag)
        @warn "The variance of the final variance is not invertible. Correction to the smallest eigenvalue(s) applied"
    end
end

function _compute_vcov(form::Misspecified, H, G, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
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


function ipinv(A::AbstractMatrix{T}; atol::Real=0.0, rtol::Real=(eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol)) where {T}
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
