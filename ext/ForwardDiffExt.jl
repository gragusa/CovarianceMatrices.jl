module ForwardDiffExt

using ForwardDiff, CovarianceMatrices, LinearAlgebra

## Defines methods for ForwardDiff dual numbers

function CovarianceMatrices.Λ!(L::AbstractVector{F}, j::Integer,
        m::AbstractMatrix{F}) where {F <: ForwardDiff.Dual}
    T, p = size(m)
    fill!(L, zero(F))

    inv_T = 1 / T
    scale = sqrt(2 * inv_T)

    @inbounds for t in 1:T
        w = cos(π * j * (t - 0.5) * inv_T)
        # Use BLAS axpy: L = L + w * m[t, :]
        #BLAS.axpy!(w, view(m, t, :), L)
        L .= L + w * view(m, t, :)
    end

    # Scale in-place
    lmul!(scale, L)
    return L
end

function CovarianceMatrices.avar(k::EWC, X::AbstractMatrix{F}; prewhite = false) where {F <: ForwardDiff.Dual}
    if prewhite
        Z, D = fit_var(X)
    else
        Z = X
    end

    B = k.B
    T, p = size(Z)
    Ω = zeros(F, p, p)
    L = Vector{F}(undef, p)

    inv_B = 1 / B

    @inbounds for j in 1:B
        CovarianceMatrices.Λ!(L, j, Z)
        # Use BLAS syr! for symmetric rank-1 update: Ω += L * L'
        # syr!(uplo, alpha, x, A) computes A = A + alpha * x * x'
        #BLAS.syr!('U', inv_B, L, Ω)
        Ω += inv_B * L * L'
    end

    # Fill lower triangle from upper (syr! only updates one triangle)
    # @inbounds for i in 1:p, j in 1:i-1
    #     Ω[i, j] = Ω[j, i]
    # end

    return Ω
end

end
