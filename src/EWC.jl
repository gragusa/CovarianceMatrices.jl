Base.@propagate_inbounds function Λ(
        j::Integer,
        m::AbstractMatrix{F}
) where {F <: AbstractFloat}
    T, p = size(m)
    L = similar(m, (p, 1))
    fill!(L, zero(F))
    for t in 1:T
        w = cos((π * j * ((t - 0.5) / T)))
        z = view(m, t, :)
        L .= L .+ w .* z
    end
    return L *= sqrt(2 / T)
end

Base.@propagate_inbounds function avar(
        k::EWC,
        X::Matrix{F};
        prewhite = false
) where {F <: AbstractFloat}
    if prewhite
        Z, D = fit_var(X)
    else
        Z = X
    end
    B = k.B
    T, p = size(Z)
    Ω = similar(Z, (p, p))
    fill!(Ω, zero(F))
    for j in 1:B
        L = Λ(j, Z)
        @. Ω += L * L'
    end
    @. Ω /= B
    return Symmetric(Ω)
end
