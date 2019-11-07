"""
    Γ(A::AbstractMatrix{T}, j::Int) where T

Calculate the autocovariance of order `j` of `A`.

# Arguments
- `A::AbstractMatrix{T}`: the matrix whose autocorrelation need to be calculated
- `j::Int`: the autocorrelation order

# Returns
- `AbstractMatrix{T}`: the autocovariance
"""
function Γ(A::Matrix{T}, j::Int) where T
    n, p = size(A)
    Q = zeros(T, p, p)
    Γsign!(Q, A, j, Val{j>0})
    return Q
end

function Γsign!(Q, A, j::Int, ::Type{Val{true}})
    n, p = size(A)
    for h=1:p, s = 1:h
        for t = j+1:n
            @inbounds Q[s, h] = Q[s, h] + A[t, s]*A[t-j, h]
        end
    end
end

function Γsign!(Q, A, j::Int, ::Type{Val{false}})
    n, p = size(A)
    for h=1:p, s = 1:h
        for t = -j+1:n
            @inbounds Q[s,h] = Q[s ,h] + A[t+j, s]*A[t,h]
        end
    end
end

covindices(k::T, n) where T<:QuadraticSpectralKernel = Iterators.filter(x -> x!=0, -n:n)
covindices(k::HAC, n) = Iterators.filter(x -> x!=0, -floor(Int, k.bw[1]):floor(Int, k.bw[1]))

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------
kernel(k::TruncatedKernel, x::Real) = (abs(x)<=1) ? one(x) : zero(x)
kernel(k::BartlettKernel, x::Real) = (abs(x)<=1) ? (one(x)-abs(x)) : zero(x)
kernel(k::TukeyHanningKernel, x::Real) = (abs(x)<=1) ? one(x)/2*(one(x)+cospi(x)) : zero(x)

function kernel(k::ParzenKernel, x::Real)
    ax = abs(x)
    return  ax <= 1/2 ? one(x)-6*ax^2+6*ax^3 : 2*one(x)*(1-ax)^3
end

function kernel(k::QuadraticSpectralKernel, x::Real)
    z = one(x)*6/5*π*x
    return 3*(sin(z)/z-cos(z))*(1/z)^2
end

function setupkernelweights!(k::HAC, m::AbstractMatrix)
    n, p = size(m)
    kw = kernelweights(k)
    if isempty(kw) || length(kw) != p || all(iszero.(kw))
        resize!(kw, p)
        idx = map(x->allequal(x), eachcol(m))
        kw .= 1.0 .- idx
    end
    return kw
end

# -----------------------------------------------------------------------------
# Optimal bandwidth
# -----------------------------------------------------------------------------
"""
    optimalbandwidth(k::HAC{T}, mm; prewhite::Bool=false) where {T<:Andrews}
    optimalbandwidth(k::HAC{T}, mm; prewhite::Bool=false) where {T<:NeweyWest}


Calculate the optimal bandwidth according to either Andrews or Newey-West.
"""
function optimalbandwidth(
    k::HAC{T},
    m::AbstractMatrix;
    prewhite::Bool
) where T<:Union{Andrews, NeweyWest}
    setupkernelweights!(k, m)
    bw = _optimalbandwidth(k, m, prewhite)
    return bw
end

optimalbandwidth(k::HAC{<:Fixed}, m::AbstractMatrix; kwargs...) = first(k.bw)

_optimalbandwidth(k::HAC{T}, mm, prewhite::Bool=false) where {T<:NeweyWest} = bwNeweyWest(k, mm, prewhite)
_optimalbandwidth(k::HAC{T}, mm, prewhite::Bool=false) where {T<:Andrews} = bwAndrews(k, mm, prewhite)
_optimalbandwidth(k::HAC{T}, mm, prewhite::Bool=false) where {T<:Fixed} = first(k.bw)

function bwAndrews(k::HAC, mm, prewhite::Bool)
    n, p  = size(mm)
    a1, a2 = getalpha(k, mm)
    k.bw[1] = bw_andrews(k, a1, a2, n)
    return k.bw[1]
end

function bwNeweyWest(k::HAC, mm, prewhite::Bool)
    w = kernelweights(k)
    bw = bandwidth(k)
    n, p = size(mm)
    l = getrates(k, mm, prewhite)
    xm = mm*w
    a = map(j -> dot(xm[1:n-j], xm[j+1:n])/n, 0:l)::Array{Float64, 1}
    aa = view(a, 2:l+1)
    a0 = a[1] + 2*sum(aa)
    a1 = 2*sum((1:l) .* aa)
    a2 = 2*sum((1:l).^2 .* aa)
    bw[1] = bwnw(k, a0, a1, a2)*(n+prewhite)^growthrate(k)
    return bw[1]
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*n)^(0.2)),
                    :BartlettKernel          => :(1.1447*(a1*n)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*n)^(0.2)),
                    :TukeyHanningKernel      => :(1.7462*(a2*n)^(0.2)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*n)^(0.2)))

for kerneltype in kernels
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha(k, mm)
    w = k.weights
    rho, σ⁴ = fit_ar(mm)
    #@show rho, σ⁴
    nm = 4.0.*(rho.^2).*σ⁴./(((1.0.-rho).^6).*((1.0.+rho).^2))
    dn = σ⁴./(1.0.-rho).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.0.*(rho.^2).*σ⁴./((1.0.-rho).^8)
    α₂ = sum(w.*nm)/sum(w.*dn)
    return α₁, α₂
end

function getrates(k, mm, prewhite::Bool)
    n, p = size(mm)
    lrate = lagtruncation(k)
    adj = prewhite ? 3 : 4
    floor(Int, adj*(n/100)^lrate)
end

@inline bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447*((s1/s0)^2)^growthrate(k)
@inline bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614*((s2/s0)^2)^growthrate(k)
@inline bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221*((s2/s0)^2)^growthrate(k)

## --> Newey-West Optimal bandwidth <---
@inline growthrate(k::HAC) = 1/5
@inline growthrate(k::BartlettKernel) = 1/3

@inline lagtruncation(k::BartlettKernel) = 2/9
@inline lagtruncation(k::ParzenKernel) = 4/25
@inline lagtruncation(k::QuadraticSpectralKernel) = 2/25


# TODO: move this function to util
function allequal(x)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i=2:length(x)
        x[i] == e1 || return false
    end
    return true
end

# -----------------------------------------------------------------------------
# Fit function
# -----------------------------------------------------------------------------
Base.@propagate_inbounds function fit_var(A::Matrix{T}) where T
    n, p = size(A)
    Y = view(A, 2:n,:)
    X = view(A, 1:n-1,:)
    B = X\Y
    E = Y - X*B
    E, B
end

Base.@propagate_inbounds function fit_ar(A::Matrix{T}) where T
    ## Estimate
    ##
    ## y_{t,j} = ρ y_{t-1,j} + ϵ
    n, p = size(A)
    rho = Vector{T}(undef, p)
    σ⁴ = similar(rho)
    for j in 1:p
        y = A[2:n, j]
        x = A[1:n-1, j]
        allequal(x) && (rho[j] = 0; σ⁴[j] = 0; continue)
        x .= x .- mean(x)
        y .= y .- mean(y)
        rho[j] = sum(x.*y)/sum(abs2, x)
        x .= x.*rho[j]
        y .= y .- x
        σ⁴[j]  = (sum(abs2, y)/(n-1))^2
    end    
    return rho, σ⁴
end

# -----------------------------------------------------------------------------
# Prewhiter and dewhiter functions    
# -----------------------------------------------------------------------------
prewhiter(mm, ::Type{Val{false}}) = (mm, similar(mm, (0,0)))
prewhiter(mm, ::Type{Val{true}}) = fit_var(mm)
dewhiter!(V, mm, D, ::Type{Val{false}}) = V
function dewhiter!(V, mm, D, ::Type{Val{true}})
    v = inv(I-D')
    V .= v*V*v'
    return V
end