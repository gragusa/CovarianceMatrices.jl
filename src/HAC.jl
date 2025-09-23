function avar(k::K, X::AbstractMatrix{F}; prewhite = false) where {K<:HAC,F<:Real}
    Z, D = finalize_prewhite(X, Val(prewhite))
    T, p = size(Z)
    wlock = k.wlock[1]
    setkernelweights!(k, X)
    k.bw .= _optimalbandwidth(k, Z, prewhite)
    k.wlock .= wlock
    V = zeros(F, p, p)
    Q = similar(V)
    kernelestimator!(k, V, Q, Z)
    v = inv(one(F)*I - D')
    return v * V * v'
end

finalize_prewhite(X, ::Val{true}) = fit_var(X)
finalize_prewhite(X, ::Val{false}) = X, ZeroMat()

struct ZeroMat end
Base.:-(J::UniformScaling, Z::ZeroMat) = J
Base.:+(J::UniformScaling, Z::ZeroMat) = J
LinearAlgebra.adjoint(Z::ZeroMat) = Z

function kernelestimator!(k::K, V::AbstractMatrix{F}, Q, Z) where {K<:HAC,F<:Real}
    ## V is the final variance
    ## Q is the temporary matrix
    ## Z is the data matrix
    ## κ is the kernel vector
    T, _ = size(Z)
    idx = covindices(k, T)
    bw = convert(F, k.bw[1])
    ## Calculate the variance at lag 0
    mul!(Q, Z', Z)
    copy!(V, Q)
    ## Calculate Γ₁, Γ₂, ..., Γⱼ
    @inbounds for j ∈ eachindex(idx)
        Zₜ = view(Z, 1:(T-j), :)
        Zₜ₊₁ = view(Z, (1+j):T, :)
        mul!(Q, Zₜ', Zₜ₊₁)
        κ = kernel(k, j/bw)
        @. V += κ * Q
        @. V += κ * Q'
    end
    return V ./ T
end

avarscaler(K::HAC, X; prewhite = false) = size(X, 1)

covindices(k::T, n) where {T<:QuadraticSpectral} = 1:n
covindices(k::T, n) where {T<:Bartlett} = 1:(floor(Int, k.bw[1]))
covindices(k::HAC, n) = 1:floor(Int, k.bw[1])
covindices(k::T, n) where {T<:HR} = 1:0
# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------
kernel(k::Truncated, x::Real) = (abs(x) <= 1) ? one(x) : zero(x)
kernel(k::Bartlett, x::Real) = (abs(x) < 1) ? (one(x) - abs(x)) : zero(x)
function kernel(k::TukeyHanning, x::Real)
    return (abs(x) <= 1) ? one(x) / 2 * (one(x) + cospi(x)) : zero(x)
end

function kernel(k::Parzen, x::Real)
    ax = abs(x)
    return ax <= 1 / 2 ? one(x) - 6 * ax^2 + 6 * ax^3 : 2 * one(x) * (1 - ax)^3
end

function kernel(k::QuadraticSpectral, x::Real)
    z = one(x) * 6 / 5 * π * x
    return 3 * (sin(z) / z - cos(z)) * (1 / z)^2
end

function setkernelweights!(k::HAC{T}, X) where {T<:Union{Andrews,NeweyWest}}
    if k.wlock[1]
        @assert length(k.kw) == size(X, 2) "The number of columns in X must match the number of kernel weights instead $(k.kw)"
    else
        resize!(k.kw, size(X, 2))
        k.kw .= 1.0 .- map(x -> CovarianceMatrices.allequal(x), eachcol(X))
    end
    return k.kw
end

setkernelweights!(k::HAC{T}, X) where {T<:Fixed} = nothing
setkernelweights!(k::AVarEstimator, X) = nothing
# -----------------------------------------------------------------------------
# Optimal bandwidth
# -----------------------------------------------------------------------------
function workingoptimalbw(
    k::HAC{T},
    m::AbstractMatrix;
    prewhite::Bool = false,
) where {T<:Union{Andrews,NeweyWest}}
    X, D = prewhiter(m, prewhite)
    setkernelweights!(k, X)
    bw = _optimalbandwidth(k, X, prewhite)
    return X, D, bw
end

function workingoptimalbw(k::HAC{T}, m::AbstractMatrix; kwargs...) where {T<:Fixed}
    return (m, Matrix{eltype{m}}(undef, 0, 0), first(k.bw))
end

"""
optimalbandwidth(k::HAC{T}, mm; prewhite::Bool=false) where {T<:Andrews}
optimalbandwidth(k::HAC{T}, mm; prewhite::Bool=false) where {T<:NeweyWest}

Calculate the optimal bandwidth according to either Andrews or Newey-West.
"""
function optimalbw(
    k::HAC{T},
    m::AbstractMatrix;
    demean::Bool = false,
    dims::Int = 1,
    means::Union{Nothing,AbstractArray} = nothing,
    prewhite::Bool = false,
) where {T<:Union{Andrews,NeweyWest}}
    X = demean ? demeaner(m; means = means, dims = dims) : m
    _, _, bw = workingoptimalbw(k, X; prewhite = prewhite)
    return bw
end

function _optimalbandwidth(k::HAC{T}, mm, prewhite) where {T<:NeweyWest}
    return bwNeweyWest(k, mm, prewhite)
end

_optimalbandwidth(k::HAC{T}, mm, prewhite) where {T<:Andrews} = bwAndrews(k, mm, prewhite)
_optimalbandwidth(k::HAC{T}, mm, prewhite) where {T<:Fixed} = first(k.bw)

function bwAndrews(k::HAC, mm, prewhite::Bool)
    n, p = size(mm)
    a1, a2 = getalpha(k, mm)
    k.bw[1] = bw_andrews(k, a1, a2, n)
    return k.bw[1]
end

function bwNeweyWest(k::HAC, mm, prewhite::Bool)
    bw = bandwidth(k)
    w = k.kw
    n, _ = size(mm)
    l = getrates(k, mm, prewhite)
    xm = mm * w
    a = Vector{eltype(xm)}(undef, l + 1)
    @inbounds for j ∈ 0:l
        a[j+1] =
            dot(
                view(xm, firstindex(xm):(lastindex(xm)-j)),
                view(xm, (j+firstindex(xm)):lastindex(xm)),
            ) / n
    end
    aa = view(a, 2:(l+1))
    a0 = a[1] + 2 * sum(aa)
    a1 = 2 * sum((1:l) .* aa)
    a2 = 2 * sum((1:l) .^ 2 .* aa)
    bw[1] = bwnw(k, a0, a1, a2) * (n + prewhite)^growthrate(k)
    return bw[1]
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(
    :Truncated => :(0.6611 * (a2 * n)^(0.2)),
    :Bartlett => :(1.1447 * (a1 * n)^(1 / 3)),
    :Parzen => :(2.6614 * (a2 * n)^(0.2)),
    :TukeyHanning => :(1.7462 * (a2 * n)^(0.2)),
    :QuadraticSpectral => :(1.3221 * (a2 * n)^(0.2)),
)

for kerneltype ∈ kernels
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha(k, mm)
    w = k.kw
    rho, σ⁴ = fit_ar(mm)
    nm = 4.0 .* (rho .^ 2) .* σ⁴ ./ (((1.0 .- rho) .^ 6) .* ((1.0 .+ rho) .^ 2))
    dn = σ⁴ ./ (1.0 .- rho) .^ 4
    α₁ = sum(w .* nm) / sum(w .* dn)
    nm = 4.0 .* (rho .^ 2) .* σ⁴ ./ ((1.0 .- rho) .^ 8)
    α₂ = sum(w .* nm) / sum(w .* dn)
    return α₁, α₂
end

function getrates(k, mm, prewhite::Bool)
    n, _ = size(mm)
    lrate = lagtruncation(k)
    adj = prewhite ? 3 : 4
    return floor(Int, adj * ((n + prewhite) / 100)^lrate)
end

@inline bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447 * ((s1 / s0)^2)^growthrate(k)
@inline bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614 * ((s2 / s0)^2)^growthrate(k)
@inline bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221 * ((s2 / s0)^2)^growthrate(k)

## --> Newey-West Optimal bandwidth <---
@inline growthrate(k::HAC) = 1 / 5
@inline growthrate(k::BartlettKernel) = 1 / 3

@inline lagtruncation(k::BartlettKernel) = 2 / 9
@inline lagtruncation(k::ParzenKernel) = 4 / 25
@inline lagtruncation(k::QuadraticSpectralKernel) = 2 / 25

function allequal(x)
    lx = length(x)
    lx < 2 && return true
    e1 = x[1]
    @inbounds for i ∈ 2:lx
        x[i] == e1 || return false
    end
    return true
end

# -----------------------------------------------------------------------------
# Fit function
# -----------------------------------------------------------------------------
Base.@propagate_inbounds function fit_var(A::AbstractMatrix{T}) where {T}
    fi = firstindex(A, 1)
    li = lastindex(A, 1)
    Y = view(A, (fi+1):li, :)
    X = view(A, fi:(li-1), :)
    B = cholesky(X'X) \ X'Y
    E = Y - X * B
    return E, B
end


Base.@propagate_inbounds function fit_ar(Z::AbstractMatrix{T}) where {T}
    ## Estimate
    ##
    ## y_{t,j} = ρ y_{t-1,j} + ϵ
    A = parent(Z)
    n, p = size(A)
    rho = Vector{T}(undef, p)
    σ⁴ = similar(rho)
    xy = Vector{T}(undef, n - 1)
    for j ∈ axes(A, 2)
        y = A[2:lastindex(A, 1), j]
        x = A[1:(lastindex(A, 1)-1), j]
        allequal(x) && (rho[j] = 0; σ⁴[j] = 0; continue)
        x .= x .- mean(x)
        y .= y .- mean(y)
        xy .= x .* y
        rho[j] = sum(xy) / sum(abs2, x)
        x .= x .* rho[j]
        y .= y .- x
        σ⁴[j] = (sum(abs2, y) / (n - 1))^2
    end
    return rho, σ⁴
end

# -----------------------------------------------------------------------------
# Prewhiter
# -----------------------------------------------------------------------------
function prewhiter(M::AbstractMatrix{T}, prewhite::Bool) where {T<:Real}
    if prewhite
        return fit_var(M)
    else
        if eltype(M) ∈ (Float32, Float64)
            return (M::Matrix{T}, Matrix{T}(undef, 0, 0))
        else
            return (float(M), zeros(0, 0))
        end
    end
end
