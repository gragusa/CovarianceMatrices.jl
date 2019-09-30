check_cache_consistenty(k::HAC, cache::HACCache{T}) where T<:Prewhiten = isprewhiten(k) ? nothing : error("Inconstent cache type")
check_cache_consistenty(k::HAC, cache::HACCache{T}) where T<:Unwhiten = !isprewhiten(k) ? nothing : error("Inconstent cache type")

Optimal() = Optimal{Andrews}()

TruncatedKernel(;prewhiten=false) = TRK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
BartlettKernel(;prewhiten=false) = BTK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
ParzenKernel(;prewhiten=false) = PRK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
TukeyHanningKernel(;prewhiten=false) = THK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
QuadraticSpectralKernel(;prewhiten=false) = QSK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)

BartlettKernel(x::Type{NeweyWest};prewhiten=false) = BTK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
ParzenKernel(x::Type{NeweyWest};prewhiten=false) = PRK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
QuadraticSpectralKernel(x::Type{NeweyWest};prewhiten=false) = QSK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
TukeyHanningKernel(x::Type{NeweyWest};prewhiten=false) = error("Newey-West optimal bandwidth does not support TukeyHanningKernel")
TruncatedKernel(x::Type{NeweyWest};prewhiten=false) = error("Newey-West optimal bandwidth does not support TuncatedKernel")

TruncatedKernel(x::Type{Andrews};prewhiten=false) = TRK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
BartlettKernel(x::Type{Andrews};prewhiten=false) = BTK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
ParzenKernel(x::Type{Andrews};prewhiten=false) = PRK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
TukeyHanningKernel(x::Type{Andrews};prewhiten=false) = THK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)
QuadraticSpectralKernel(x::Type{Andrews};prewhiten=false) = QSK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0), prewhiten)

TruncatedKernel(bw::Number;prewhiten=false) = TRK(Fixed(), [float(bw)], Array{WFLOAT}(undef,0), prewhiten)
BartlettKernel(bw::Number;prewhiten=false) = BTK(Fixed(), [float(bw)], Array{WFLOAT}(undef,0), prewhiten)
ParzenKernel(bw::Number;prewhiten=false) = PRK(Fixed(), [float(bw)], Array{WFLOAT}(undef,0), prewhiten)
TukeyHanningKernel(bw::Number;prewhiten=false) = THK(Fixed(), [float(bw)], Array{WFLOAT}(undef,0), prewhiten)
QuadraticSpectralKernel(bw::Number;prewhiten=false) = QSK(Fixed(), [float(bw)], Array{WFLOAT}(undef,0), prewhiten)

# bandwidth(k::HAC{G}, X::AbstractMatrix) where {G<:Fixed} = k.bw
# bandwidth(k::HAC{Optimal{G}}, X::AbstractMatrix) where {G<:Andrews} = bwAndrews(k, )

# function bandwidth(k::QuadraticSpectralKernel, X::AbstractMatrix)
#     return k.bw(X, k)
# end

isprewhiten(k::HAC) = k.prewhiten

getcovindeces(k::T, n) where T<:QuadraticSpectralKernel = Iterators.filter(x -> x!=0, -n:n)
getcovindeces(k::HAC, n) = Iterators.filter(x -> x!=0, -floor(Int, k.bw[1]):floor(Int, k.bw[1]))

function Γ!(cache, j)
    X = cache.XX
    T, p = size(X)
    Q = fill!(cache.Q, zero(eltype(X)))
    if j >= 0
        for h=1:p, s = 1:h
            for t = j+1:T
                @inbounds Q[s, h] = Q[s, h] + X[t, s]*X[t-j, h]
            end
        end
    elseif j<0
        for h=1:p, s = 1:h
            for t = -j+1:T
                @inbounds Q[s,h] = Q[s ,h] + X[t+j, s]*X[t,h]
            end
        end
    end
    return cache.Q
end

prewhiten!(cache::HACCache{T}) where T<:Unwhiten = copyto!(cache.XX, cache.q)
prewhiten!(cache::HACCache{T}) where T<:Prewhiten = fit_var!(cache)
swhiten!(cache::HACCache{T}) where T<:Unwhiten = nothing

function swhiten!(cache::HACCache{T}) where T<:Prewhiten
    fill!(cache.Q, zero(eltype(cache.Q)))
    for i = 1:size(cache.Q, 2)
        cache.Q[i,i] = one(eltype(cache.Q))
    end
    v = ldiv!(qr(I-cache.D'), cache.Q)
    cache.V .= v*cache.V*v'
end

##############################################################################
##
## Kernel methods
##
##############################################################################

kernel(k::HAC, x::Real) = isnan(x) ? (return 1.0) : kernel(k, float(x))
kernel(k::TruncatedKernel, x::Real)    = (abs(x) <= 1.0) ? 1.0 : 0.0
kernel(k::BartlettKernel, x::Real)     = (abs(x) <= 1.0) ? (1.0 - abs(x)) : 0.0
kernel(k::TukeyHanningKernel, x::Real) = (abs(x) <= 1.0) ? 0.5 * (1.0 + cospi(x)) : 0.0

function kernel(k::ParzenKernel, x::Real)
    ax = abs(x)
    if ax <= 0.5
        1.0 - 6.0 * ax^2 + 6.0 * ax^3
    else
        2.0 * (1.0 - ax)^3
    end
end

function kernel(k::QuadraticSpectralKernel, x::Real)
    iszero(x) ? 1.0 : (z = 1.2*π*x; 3*(sin(z)/z-cos(z))*(1/z)^2)
end

function setupkernelweights!(k, p, xtype)
    if isempty(k.weights)
        for j in 1:p
            push!(k.weights, one(xtype))
        end
    elseif all(iszero.(k.weights))
        fill!(k.weights, one(xtype))
    end
end

##############################################################################
##
## Fit functions
##
##############################################################################

 function fit_var!(cache::HACCache)
    #X, Y, Z, u, D = cache.XX, cache.YY, cache.q, cache.u, cache.D
    XX, YY, q, u, D = cache.XX, cache.YY, cache.q, cache.u, cache.D

    n, p = size(q)
    @inbounds for j in 1:p, i = 1:n-1
        XX[i,j] = q[i,  j]
        YY[i,j] = q[i+1,j]
    end
    QX = qr(XX)
    ldiv!(D, QX, convert(Matrix{eltype(QX)}, YY))
    @inbounds for j in 1:p, i = 1:n-1
        YY[i,j] = q[i+1,j]
    end
    mul!(u, XX, D)
    broadcast!(-, XX, YY, u)
 end

 function fit_ar!(cache)
     ## Estimate
     ##
     ## y_{t,j} = ρ y_{t-1,j} + ϵ
     σ⁴ = cache.σ⁴
     ρ = cache.ρ
     U = cache.U
     n, p = size(cache.XX)
     lag!(cache)
     Y = cache.YL
     X = cache.XL
     for j in 1:p
         y = view(Y, :, j)
         x = view(X, :, j)
         x .= x .- mean(x)
         y .= y .- mean(y)
         ρ[j] = sum(broadcast!(*, cache.U, x, y))/sum(abs2, x)
         copyto!(U, y)
         x .= x.*ρ[j]
         broadcast!(-, U, U, x)
         σ⁴[j]  = (dot(U, U)/(n-1))^2
     end
 end

 function lag!(cache)
     nl, pl = size(cache.YL)
     n, p  = size(cache.XX)
     for ic in 1:p
         for i = 2:n
             @inbounds cache.YL[i-1, ic] = cache.XX[i, ic]
             @inbounds cache.XL[i-1, ic] = cache.XX[i-1, ic]
         end
     end
  end

##############################################################################
##
## Optimal bandwidth
##
##############################################################################
optimal_bw!(cache, k::HAC, optype::T) where T<:NeweyWest = bwNeweyWest(cache, k)
optimal_bw!(cache, k::HAC, opttype::T) where T<:Andrews = bwAndrews(cache, k)

function bwAndrews(cache, k::HAC)
    isempty(k.weights) && (fill!(k.weights, 1.0))
    n, p  = size(cache.XX)
    a1, a2 = getalpha!(cache, k.weights)
    k.bw[1] = bw_andrews(k, a1, a2, n)
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*n)^(0.2)),
                    :BartlettKernel          => :(1.1447*(a1*n)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*n)^(0.2)),
                    :TukeyHanningKernel      => :(1.7462*(a2*n)^(0.2)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*n)^(0.2)))

for kerneltype in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel]
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha!(cache, w)
    fit_ar!(cache)
    σ⁴, ρ = cache.σ⁴, cache.ρ
    nm = 4.0.*(ρ.^2).*σ⁴./(((1.0.-ρ).^6).*((1.0.+ρ).^2))
    dn = σ⁴./(1.0.-ρ).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.0.*(ρ.^2).*σ⁴./((1.0.-ρ).^8)
    α₂ = sum(w.*nm)/sum(w.*dn)
    return α₁, α₂
end

function bwNeweyWest(cache, k::HAC)
    n, p = size(cache.XX)
    l = getrates(cache, k)
    w = k.weights
    xm = cache.XX*w
    a = map(j -> dot(xm[1:n-j], xm[j+1:n])/n, 0:l)::Array{Float64, 1}
    aa = view(a, 2:l+1)
    a0 = a[1] + 2*sum(aa)
    a1 = 2*sum((1:l) .* aa)
    a2 = 2*sum((1:l).^2 .* aa)
    k.bw[1] = bwnw(k, a0, a1, a2)*(n+isprewhiten(k))^growthrate(k)
end

function getrates(cache, k)
    n, p = size(cache.q)
    lrate = lagtruncation(k)
    adj = isprewhiten(k) ? 3 : 4
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
