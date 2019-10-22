Optimal() = Optimal{Andrews}()

TruncatedKernel() = TRK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
BartlettKernel() = BTK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
ParzenKernel() = PRK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
TukeyHanningKernel() = THK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
QuadraticSpectralKernel() = QSK(Optimal(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))

BartlettKernel(x::Type{NeweyWest}) = BTK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
ParzenKernel(x::Type{NeweyWest}) = PRK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
QuadraticSpectralKernel(x::Type{NeweyWest}) = QSK(Optimal{NeweyWest}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
TukeyHanningKernel(x::Type{NeweyWest}) = error("Newey-West optimal bandwidth does not support TukeyHanningKernel")
TruncatedKernel(x::Type{NeweyWest}) = error("Newey-West optimal bandwidth does not support TuncatedKernel")

TruncatedKernel(x::Type{Andrews}) = TRK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
BartlettKernel(x::Type{Andrews}) = BTK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
ParzenKernel(x::Type{Andrews}) = PRK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
TukeyHanningKernel(x::Type{Andrews}) = THK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))
QuadraticSpectralKernel(x::Type{Andrews}) = QSK(Optimal{Andrews}(), Array{WFLOAT}(undef,1), Array{WFLOAT}(undef,0))

TruncatedKernel(bw::Number) = TRK(Fixed(), Array{WFLOAT}(undef, 1), Array{WFLOAT}(undef,0))
BartlettKernel(bw::Number) = BTK(Fixed(), Array{WFLOAT}(undef, 1), Array{WFLOAT}(undef,0))
ParzenKernel(bw::Number) = PRK(Fixed(), Array{WFLOAT}(undef, 1), Array{WFLOAT}(undef,0))
TukeyHanningKernel(bw::Number) = THK(Fixed(), Array{WFLOAT}(undef, 1), Array{WFLOAT}(undef,0))
QuadraticSpectralKernel(bw::Number) = QSK(Fixed(), Array{WFLOAT}(undef, 1), Array{WFLOAT}(undef,0))

covindices(k::T, n) where T<:QuadraticSpectralKernel = Iterators.filter(x -> x!=0, -n:n)
covindices(k::HAC, n) = Iterators.filter(x -> x!=0, -floor(Int, k.bw[1]):floor(Int, k.bw[1]))

function Γ(A::Matrix{T}, j) where T
    n, p = size(A)
    Q = zeros(T, p, p)
    Γsign!(Q, A, Val{j>0})
    return Q
end

function Γsign!(Q, A, ::Type{Val{true}})
    for h=1:p, s = 1:h
        for t = j+1:n
            @inbounds Q[s, h] = Q[s, h] + A[t, s]*A[t-j, h]
        end
    end
end

function Γsign!(Q, A, ::Type{Val{false}})
    for h=1:p, s = 1:h
        for t = -j+1:n
            @inbounds Q[s,h] = Q[s ,h] + A[t+j, s]*A[t,h]
        end
    end
end

function getbandwidth(k::HAC{Optimal{T}}, m::AbstractMatrix{F}) where {T<:OptimalBandwidth,F}
    setupkernelweights!(k, m)
    bw = optimal_bw(k, T(), m)
    return bw
end

getbandwidth(k::HAC{T}, m::AbstractMatrix) where {T<:Fixed} = first(k.bw)

##############################################################################
##
## Kernel methods
##
##############################################################################

#kernel(k::HAC, x::Real) = isnan(x) ? (return 1.0) : kernel(k, float(x))
kernel(k::TruncatedKernel, x::Real)    = (abs(x) <= 1) ? 1 : 0
kernel(k::BartlettKernel, x::Real)     = (abs(x) <= 1.0) ? (1 - abs(x)) : 0
kernel(k::TukeyHanningKernel, x::Real) = (abs(x) <= 1.0) ? 0.5 * (1.0 + cospi(x)) : 0.0

function kernel(k::ParzenKernel, x::Real)
    ax = abs(x)
    if ax <= 1/2
        1 - 6 * ax^2 + 6 * ax^3
    else
        2 * (1 - ax)^3
    end
end

function kernel(k::QuadraticSpectralKernel, x::Real)
    iszero(x) ? 1.0 : (z = 1.2*π*x; 3*(sin(z)/z-cos(z))*(1/z)^2)
end

function setupkernelweights!(k, m::AbstractMatrix)
    n, p = size(m)
    ## k.weights is WFLOAT
    if isempty(k.weights)
        for j in 1:p
            push!(k.weights, one(WFLOAT))
        end
    elseif all(iszero.(k.weights))
        fill!(k.weights, one(WFLOAT))
    end
end

##############################################################################
##
## Fit functions
##
##############################################################################
function fit_var(A::Matrix{T}) where T
    n, p = size(A)
    Y = view(A, 2:n,:)
    X = view(A, 1:n-1,:)
    B = X\Y
    E = Y - X*B
    E, B
end

function fit_ar(A::Matrix{T}) where T
    ## Estimate
    ##
    ## y_{t,j} = ρ y_{t-1,j} + ϵ
    n, p = size(A)
    rho = Vector{T}(undef, p)
    σ⁴ = similar(rho)
    for j in 1:p
        y = A[2:n, j]
        x = A[1:n-1, j]
        x .= x .- mean(x)
        y .= y .- mean(y)
        rho[j] = sum(x.*y)/sum(abs2, x)
        x .= x.*rho[j]
        y .= y .- x
        σ⁴[j]  = (sum(abs2, y)/(n-1))^2
    end
    return rho, σ⁴
end

##############################################################################
##
## Optimal bandwidth
##
##############################################################################
optimal_bw(k::HAC, optype::T, mm; prewhite::Bool=false) where T<:NeweyWest = bwNeweyWest(k, mm, prewhite)
optimal_bw(k::HAC, opttype::T, mm; prewhite::Bool=false) where T<:Andrews = bwAndrews(k, mm, prewhite)

function bwAndrews(k::HAC, mm, prewhite)
    isempty(k.weights) && (fill!(k.weights, 1.0))
    n, p  = size(mm)
    a1, a2 = getalpha(k, mm)
    k.bw[1] = bw_andrews(k, a1, a2, n)
    return k.bw[1]
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*n)^(1/5)),
                    :BartlettKernel          => :(1.1447*(a1*n)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*n)^(1/5)),
                    :TukeyHanningKernel      => :(1.7462*(a2*n)^(1/5)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*n)^(1/5)))

for kerneltype in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel]
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha(k, mm)
    w = k.weights
    rho, σ⁴ = fit_ar(mm)
    ##rho2 = (rho.^2)
    nm = 4.0.*(rho.^2).*σ⁴./(((1.0.-rho).^6).*((1.0.+rho).^2))
    dn = σ⁴./(1.0.-rho).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.0.*(rho.^2).*σ⁴./((1.0.-rho).^8)
    α₂ = sum(w.*nm)/sum(w.*dn)
    return α₁, α₂
end

function bwNeweyWest(k::HAC, mm, prewhite)
    n, p = size(mm)
    l = getrates(k, mm, prewhite)
    w = k.weights
    xm = mm*w
    a = map(j -> dot(xm[1:n-j], xm[j+1:n])/n, 0:l)::Array{Float64, 1}
    aa = view(a, 2:l+1)
    a0 = a[1] + 2*sum(aa)
    a1 = 2*sum((1:l) .* aa)
    a2 = 2*sum((1:l).^2 .* aa)
    k.bw[1] = bwnw(k, a0, a1, a2)*(n+prewhite)^growthrate(k)
end

function getrates(k, mm, prewhite)
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
