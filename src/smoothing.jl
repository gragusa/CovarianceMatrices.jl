abstract type AbstractSmoother end

"""
Identity()

Construct a `Identity` Smoother.

"""
struct IdentitySmoother <: AbstractSmoother end
IdentitySmoother(args...) = IdentitySmoother()

"""
Truncate(ξ::Int)

Construct a `Truncated<:Smoother` with window half-size equal to ξ. 

Given a matrix A[i,j], its smoothed version is defined as 

``A[t, j] = \\frac{1}{S_T} \\sum_{s=max{t-T,-ξ}}^{min{t-1, ξ}} A[t-s,j]``

where ``S_T = (2\\xi+1)/2`` is the bandwidth.

"""
struct TruncatedSmoother <: AbstractSmoother
    ξ::Int
    S::WFLOAT
    κ::Array{WFLOAT,1}
    function TruncatedSmoother(S::Real)
        if S < 0
            throw(ArgumentError("The bandwidth must be positive"))
        else
            new(floor(Int, (S * 2 - 1) / 2), S, [2.0, 2.0, 2.0])
        end
    end
end

inducedkernel(x::Type{TruncatedSmoother}) = Bartlett


"""
Truncate(ξ::Int)

Construct a `Bartlett<:Smoother` with window half-size equal to ξ. 

Given a matrix A[i,j], its smoothed version is defined as 

``A[t, j] = \\frac{1}{S_T} \\sum_{s=max{t-T,-ξ}}^{min{t-1, ξ}} (1-|s/S_T|) A[t-s,j]``

where ``S_T = (2\\xi+1)/2`` is the bandwidth.
"""
struct BartlettSmoother <: AbstractSmoother
    ξ::Int
    S::WFLOAT
    κ::Array{WFLOAT,1}
    function BartlettSmoother(S::Real)
        if S < 0
            throw(ArgumentError("The bandwidth must be positive"))
        else
            new(floor(Int, (S * 2 - 1) / 2), S, [1.0, 2.0 / 3.0, 0.5])
        end
    end
end

bw(s::AbstractSmoother) = s.S
κ₁(s::AbstractSmoother) = s.κ₁
κ₂(s::AbstractSmoother) = s.κ₂
ξ(s::AbstractSmoother) = s.ξ


(k::IdentitySmoother)(G) = G

Base.@propagate_inbounds function (s::TruncatedSmoother)(G::Matrix)
    N, M = size(G)
    nG = zeros(WFLOAT, N, M)
    b = bw(s)
    xi = ξ(s)
    for m = axes(G, 2)
        for t = axes(G, 1)
            low = max((t - N), -xi)::Int
            high = min(t - 1, xi)::Int
            for s = low:high
                @inbounds nG[t, m] += G[t-s, m]
            end
        end
    end
    return nG
end

Base.@propagate_inbounds function (s::BartlettSmoother)(G::Matrix)
    N, M = size(G)
    b = bw(s)
    xi = ξ(s)
    nG = zeros(WFLOAT, N, M)
    for m = axes(G, 2)
        for t = axes(G, 1)
            low = max((t - N), -xi)::Int
            high = min(t - 1, xi)::Int
            for s = low:high
                κ = 1 - abs(s / b)
                nG[t, m] += κ * G[t-s, m]
            end
        end
    end
    return nG
end

Base.@propagate_inbounds function (s::TruncatedSmoother)(N::Int)
    b = bw(s)
    xi = ξ(s)
    nG = zeros(WFLOAT, 3)
    low = max((1 - N), -xi)::Int
    high = min(N - 1, xi)::Int
    for j in 1:3
        for s = low:high
            nG[j] += one(WFLOAT)
        end
    end
    return nG
end

Base.@propagate_inbounds function (s::BartlettSmoother)(N::Int)
    b = bw(s)
    xi = ξ(s)
    nG = zeros(WFLOAT, 3)
    low = max((1 - N), -xi)::Int
    high = min(N - 1, xi)::Int
    for j in 1:3
        for s = low:high
            κ = 1 - abs(s / b)
            nG[j] += (κ^j)
        end
    end
    return nG
end

struct Smoothed{K<:HAC}
    k::K
end

Smoothed{K}() where {K} = Smoothed{K}(K())

function lrvar(
    k::Smoothed,
    m::AbstractMatrix;
    prewhite::Bool=false,
    demean::Bool=true,
    scale::Real=1,
)
    mm = demean ? m .- mean(m, dims=1) : m
    scale *= inv(size(m, 1))
    return _lrvar(k, mm, prewhite, scale)
end

function getsmoother(k::Smoothed{T}, m, prewhite) where {T<:BartlettKernel}
    S = getsmoothedbandwidth(k.k, m, prewhite)
    sm = TruncatedSmoother(S)
    ## Caclulate κ̂
    sm.κ .= sm(size(m, 1))
    sm
end

function getsmoother(k::Smoothed{T}, m, prewhite) where {T<:ParzenKernel}
    S = getsmoothedbandwidth(k.k, m, prewhite)
    sm = BartlettSmoother(S)
    sm.κ .= sm(size(m, 1))
    sm
end

function getsmoothedbandwidth(k::T, m, prewhite) where {T}
    return optimalbandwidth(k, m; prewhite=prewhite)
end

# function _lrvar(k::Smoothed{T}, mm, prewhite::Bool, scale) where {T<:Union{BartlettKernel,ParzenKernel}}
#     n, p = size(mm)
#     m, D = prewhiter(mm, Val{prewhite})
#     smoother = getsmoother(k, m, prewhite)
#     sm = smoother(m)
#     V = (sm'sm) / smoother.κ[2]
#     dewhiter!(V, m, D, Val{prewhite})
#     return Symmetric(V .* convert(eltype(V), scale))
# end
