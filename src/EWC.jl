function Λ!(L::AbstractVector{F}, j::Integer, m::AbstractMatrix{F}) where {F <: Real}
    T, p = size(m)
    fill!(L, zero(F))

    inv_T = 1 / T
    scale = sqrt(F(2))

    @inbounds for t in 1:T
        w = cos(π * j * (t - F(0.5)) * inv_T)
        # L = L + w * m[t, :]
        axpy!(w, view(m, t, :), L)
    end
    # Scale in-place
    lmul!(scale, L)
    return L
end

# Non-mutating version
function Λ(j::Integer, m::AbstractMatrix{F}) where {F <: Real}
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

    for j in 1:B
        Λ!(L, j, Z)
        _rank1_upper!(Ω, inv_B, L)
    end
    # Fill the lower triangle from the upper one.
    for i in 1:p, j in 1:(i - 1)

        Ω[i, j] = Ω[j, i]
    end

    # Transform back if prewhitened
    v = inv(one(F) * I - D')
    return v * Ω * v'
end

# Ω += α·L·L' on the upper triangle. BLAS handles Float32 and Float64; any other
# `Real` (e.g. `ForwardDiff.Dual`) takes the generic loop.
function _rank1_upper!(Ω::StridedMatrix{F}, α, L::StridedVector{F}) where {F <: Union{
        Float32, Float64}}
    BLAS.syr!('U', F(α), L, Ω)
end

function _rank1_upper!(Ω, α, L)
    for j in eachindex(L), i in firstindex(L):j

        Ω[i, j] += α * L[i] * L[j]
    end
    return Ω
end
