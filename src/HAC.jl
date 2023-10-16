function avar(k::T, X::AbstractMatrix{F}; prewhiten=false) where {T<:HAC, F<:AbstractFloat}
  if prewhiten
    Z, D = fit_var(X)
  else
    Z = X
  end
  ## Setup kernel weights
  if (isempty(k.weights) || length(k.weights) != size(Z,2))
    resize!(k.weights, size(Z,2))
    fill!(k.weights, 1.0)
  end
  k.bw .= _optimalbandwidth(k, Z, prewhiten)
  V = _avar(k, Z)
  if prewhiten
    v = inv(I-D')
    return v*V*v'
  else
    return Symmetric(Matrix(V))
  end
end

function _avar(k::HAC, Z::Matrix{T}) where T<:AbstractFloat
  V = triu!(Z'*Z)
  F = eltype(V)
  idx = CovarianceMatrices.covindices(k, size(Z,1))
  bw = first(k.bw)
  Q = Matrix{F}(undef, size(V))
  @inbounds tril!(Q, -1)
  @inbounds for j in idx
    κⱼ = CovarianceMatrices.kernel(k, j/bw)
    CovarianceMatrices.Γplus!(Q, Z, j)    
    LinearAlgebra.axpy!(κⱼ, Q, V)
    tril!(Q, -1)
    CovarianceMatrices.Γminus!(Q, Z, -j)
    LinearAlgebra.axpy!(κⱼ, Q, V)
    tril!(Q, -1)
  end
  Symmetric(V)
end

avarscaler(K::HAC, X; prewhiten=false) = size(X, 1)

"""
Γplus!(Q, A::AbstractMatrix{T}, j::Int) where T

Calculate the autocovariance of order `|j|` of `A`.

# Arguments
- `Q`::AbstractMatrix{T}`: the matrix where the autocovariance will be stored
- `A::AbstractMatrix{T}`: the matrix whose autocorrelation need to be calculated
- `j::Int`: the autocorrelation order (must be positive)

# Returns
- `AbstractMatrix{T}`: the autocovariance (upper triangular) 
"""
function Γplus!(Q::Matrix, A::AbstractMatrix, j::Int)
  @inbounds for h in axes(A, 2), s in firstindex(A,1):h+firstindex(A,1)-1
        for t in (j+firstindex(A,1)):lastindex(A, 1)
            Q[s, h] = Q[s, h] + A[t, s]*A[t-j, h]
        end
    end
    return Q
end

"""
Γminus!(Q, A::AbstractMatrix{T}, j::Int) where T

Calculate the autocovariance of order `j` of `A`.

# Arguments
- `Q`::AbstractMatrix{T}`: the matrix where the autocovariance will be stored
- `A::AbstractMatrix{T}`: the matrix whose autocorrelation need to be calculated
- `j::Int`: the autocorrelation order (must be negative)

# Returns
- `AbstractMatrix{T}`: the autocovariance (upper triangular) 
"""
function Γminus!(Q::Matrix, A::AbstractMatrix, j::Int)
  @inbounds for h in axes(A, 2), s in firstindex(A,1):h+firstindex(A,1)-1
    for t in (-j+firstindex(A,1)):lastindex(A, 1)
      Q[s,h] = Q[s ,h] + A[t+j, s]*A[t,h]
    end
  end
  Q
end

function Γminus(A::AbstractMatrix, j::Int)
  Q = zeros(eltype(A), size(A,2), size(A,2))
  Γminus!(Q, A, j)
  return Symmetric(Q)
end

function Γplus(A::AbstractMatrix, j::Int)
  Q = zeros(eltype(A), size(A,2), size(A,2))
  Γplus!(Q, A, j)
  return Symmetric(Q)
end

covindices(k::T, n) where T<:QuadraticSpectral = 1:n
covindices(k::T, n) where T<:Bartlett = 1:(floor(Int, k.bw[1]))
covindices(k::HAC, n) = 1:floor(Int, k.bw[1])

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------
kernel(k::Truncated, x::Real) = (abs(x)<=1) ? one(x) : zero(x)
kernel(k::Bartlett, x::Real) = (abs(x)<1) ? (one(x)-abs(x)) : zero(x)
kernel(k::TukeyHanning, x::Real) = (abs(x)<=1) ? one(x)/2*(one(x)+cospi(x)) : zero(x)

function kernel(k::Parzen, x::Real)
  ax = abs(x)
  return  ax <= 1/2 ? one(x)-6*ax^2+6*ax^3 : 2*one(x)*(1-ax)^3
end

function kernel(k::QuadraticSpectral, x::Real)
  z = one(x)*6/5*π*x
  return 3*(sin(z)/z-cos(z))*(1/z)^2
end

function setkernelweights!(k::HAC, m::AbstractMatrix)
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

function workingoptimalbw(
    k::HAC{T},
    m::AbstractMatrix;
    prewhiten::Bool=false,    
    ) where T<:Union{Andrews, NeweyWest}    
    X, D = prewhiter(m, prewhiten)
    setkernelweights!(k, X)
    bw = _optimalbandwidth(k, X, prewhiten)
    return X, D, bw
end

workingoptimalbw(k::HAC{T}, m::AbstractMatrix; kwargs...) where T<:Fixed = (m, Matrix{eltype{m}}(undef,0,0), first(k.bw))

"""
optimalbandwidth(k::HAC{T}, mm; prewhiten::Bool=false) where {T<:Andrews}
optimalbandwidth(k::HAC{T}, mm; prewhiten::Bool=false) where {T<:NeweyWest}


Calculate the optimal bandwidth according to either Andrews or Newey-West.
"""
function optimalbw(
  k::HAC{T},
  m::AbstractMatrix;
  demean::Bool = false,
  dims::Int = 1,
  means::Union{Nothing, AbstractArray} = nothing,
  prewhiten::Bool=false
  ) where T<:Union{Andrews, NeweyWest}
  X = demean ? demeaner(m, means; dims=dims) : m
  _, _, bw = workingoptimalbw(k, X; prewhiten=prewhiten)
  return bw
end

_optimalbandwidth(k::HAC{T}, mm, prewhiten) where {T<:NeweyWest} = bwNeweyWest(k, mm, prewhiten)
_optimalbandwidth(k::HAC{T}, mm, prewhiten) where {T<:Andrews} = bwAndrews(k, mm, prewhiten)
_optimalbandwidth(k::HAC{T}, mm, prewhiten) where {T<:Fixed} = first(k.bw)

function bwAndrews(k::HAC, mm, prewhiten::Bool)
  n, p  = size(mm)
  a1, a2 = getalpha(k, mm)
  k.bw[1] = bw_andrews(k, a1, a2, n)
  return k.bw[1]
end

function bwNeweyWest(k::HAC, mm, prewhiten::Bool)
  w = kernelweights(k)
  bw = bandwidth(k)
  n, _ = size(mm)
  l = getrates(k, mm, prewhiten)
  xm = mm*w
  a = Vector{eltype(xm)}(undef, l+1)
  @inbounds for j in 0:l
    a[j+1] = dot(view(xm, firstindex(xm):lastindex(xm)-j), view(xm, j+firstindex(xm):lastindex(xm)))/n
  end
  aa = view(a, 2:l+1)
  a0 = a[1] + 2*sum(aa)
  a1 = 2*sum((1:l) .* aa)
  a2 = 2*sum((1:l).^2 .* aa)
  bw[1] = bwnw(k, a0, a1, a2)*(n+prewhiten)^growthrate(k)
  return bw[1]
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(
:Truncated         => :(0.6611*(a2*n)^(0.2)),
:Bartlett          => :(1.1447*(a1*n)^(1/3)),
:Parzen            => :(2.6614*(a2*n)^(0.2)),
:TukeyHanning      => :(1.7462*(a2*n)^(0.2)),
:QuadraticSpectral => :(1.3221*(a2*n)^(0.2)))

for kerneltype in kernels
  @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha(k, mm)
  w = k.weights
  rho, σ⁴ = fit_ar(mm)
  nm = 4.0.*(rho.^2).*σ⁴./(((1.0.-rho).^6).*((1.0.+rho).^2))
  dn = σ⁴./(1.0.-rho).^4
  α₁ = sum(w.*nm)/sum(w.*dn)
  nm = 4.0.*(rho.^2).*σ⁴./((1.0.-rho).^8)
  α₂ = sum(w.*nm)/sum(w.*dn)
  return α₁, α₂
end

function getrates(k, mm, prewhiten::Bool)
  n, _ = size(mm)
  lrate = lagtruncation(k)
  adj = prewhiten ? 3 : 4
  floor(Int, adj*((n+prewhiten)/100)^lrate)
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
Base.@propagate_inbounds function fit_var(A::AbstractMatrix{T}) where T
  fi = firstindex(A, 1)
  li = lastindex(A, 1) 
  Y = view(A, fi+1:li,:)
  X = view(A, fi:li-1, :)
  B = cholesky(X'X)\X'Y
  E = Y - X*B
  return E, B
end

Base.@propagate_inbounds function fit_var(A::AbstractSparseMatrix{T}) where T
  P = parent(A)
  li = lastindex(P, 1)
  Y = P[2:li,:]
  X = P[1:li-1, :]
  B = qr!(X'X)\Matrix(X'Y)
  E = Y - X*B
  return E, B
end

Base.@propagate_inbounds function fit_ar(Z::AbstractMatrix{T}) where T
  ## Estimate
  ##
  ## y_{t,j} = ρ y_{t-1,j} + ϵ
  A = parent(Z)
  n, p = size(A)
  rho = Vector{T}(undef, p)
  σ⁴ = similar(rho)
  xy = Vector{T}(undef, n-1)
  for j in axes(A, 2)
    y = A[2:lastindex(A,1), j]
    x = A[1:lastindex(A,1)-1, j]
    allequal(x) && (rho[j] = 0; σ⁴[j] = 0; continue)
    x .= x .- mean(x)
    y .= y .- mean(y)
    xy .= x.*y
    rho[j] = sum(xy)/sum(abs2, x)
    x .= x.*rho[j]
    y .= y .- x
    σ⁴[j]  = (sum(abs2, y)/(n-1))^2
  end
  return rho, σ⁴
end

# -----------------------------------------------------------------------------
# Prewhiter
# -----------------------------------------------------------------------------
function prewhiter(M::AbstractMatrix{T}, prewhiten::Bool) where T<:AbstractFloat
  if prewhiten
    return fit_var(M)
  else
    if eltype(M) ∈ (Float32, Float64)
      return (M::Matrix{T}, Matrix{T}(undef, 0, 0))
    else
      return (float(M), zeros(0, 0))
    end
  end
end
