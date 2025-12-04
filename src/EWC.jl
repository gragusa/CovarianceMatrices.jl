function Λ!(L::AbstractVector{F}, j::Integer, m::AbstractMatrix{F}) where {F <: Real}
    T, p = size(m)
    fill!(L, zero(F))

    inv_T = 1 / T
    scale = sqrt(F(2))

    @inbounds for t in 1:T
        w = cos(π * j * (t - F(0.5)) * inv_T)
        # Use BLAS axpy: L = L + w * m[t, :]
        #L .= L + w * view(m, t, :)
        BLAS.axpy!(w, view(m, t, :), L)
    end
    # Scale in-place
    lmul!(scale, L)
    return L
end

# Non-mutating version
function Λ(j::Integer, m::AbstractMatrix{F}) where {F <: AbstractFloat}
    T, p = size(m)
    L = zeros(F, p)
    return Λ!(L, j, m)
end

function avar(k::EWC, X::AbstractMatrix{F}; prewhite = false) where {F <: Real}
    Z, D = finalize_prewhite(X, Val(prewhite))

    B = k.B
    T, p = size(Z)
    Ω = zeros(F, p, p)
    L = Vector{F}(undef, p)

    inv_B = 1 / B

    @inbounds for j in 1:B
        Λ!(L, j, Z)
        # Use BLAS syr! for symmetric rank-1 update: Ω += L * L'
        # syr!(uplo, alpha, x, A) computes A = A + alpha * x * x'
        #Ω += inv_B * L * L'
        BLAS.syr!('U', inv_B, L, Ω)
    end
    # Fill lower triangle from upper (syr! only updates one triangle)
    @inbounds for i in 1:p, j in 1:(i - 1)

        Ω[i, j] = Ω[j, i]
    end

    # Transform back if prewhitened
    v = inv(one(F) * I - D')
    return v * Ω * v'
end
