function lag!{T, F}(Yl::Array{T, 2}, Y::AbstractArray{F, 1}, p::Int64)
    ## Given an Array, return the matrix
  ## of lagged values of Y
  ## [y_t-p, y_{t-p-1}, ..., y_{t-1}]
  N  = length(Y)
  Ts, pl = size(Yl)
  for i = 1:p
    for j = 1:N-p
      @inbounds Yl[j, i] = Y[p-i+j]
    end
  end
  if pl==p+1
    for j=1:N-p
      @inbounds Yl[j,p+1] = 1.0
    end
  end
end

function olsvar{T<:Number}(y::Matrix{T})
    # Input : data y, lag order 1
    # Output: coefficient estimate β, residual U
    N, K = size(y)
    Y = y[2:N,:]  # LHS variable
    X = y[1:N-1,:]
    A = X\Y    # OLS estimator
    U = Y-X*A  # estimated residuals
    return U, A'
end

function ar{T<:Number}(Y::Matrix{T}, lag::Int)
  N, p = size(Y)
  Yl = Array{Float64}(N-lag, lag+1)
  ρ  = Array{Float64}(p, lag)
  σ² = Array{Float64}(p)
  ε  = Array{Float64}(N-lag)
  for j in 1:p
    lag!(Yl, view(Y, :,j), lag)
    Yt          = view(Y, lag+1:N, j)
    θ           = Yl\Yt
    ρ[j, 1:lag] = θ[1:lag]
    ε           .= Yt - Yl*θ
    σ²[j]       = dot(ε, ε)/(N-lag)
  end
  return ρ, σ²
end

ar{T<:Number}(Y::Matrix{T}) = ar(Y, 1)

function pre_white(X::Matrix)
    X, D = olsvar(X)
    (X, inv(I-D))
end


## ---> Andrews Optimal bandwidth <---

d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*N)^(0.2)),
                    :BartlettKernel          => :(1.1447*(a1*N)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*N)^(0.2)),
                    :TukeyHanningKernel      => :(1.7462*(a2*N)^(0.2)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*N)^(0.2)))

for tty in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel]
    @eval $:(bw_andrews)(k::($tty), a1, a2, N) = $(d_bw_andrews[tty])
end

function getalpha(X::AbstractMatrix, approx::Symbol, w::Vector)
    ρ, σ² = ar(X)
    σ⁴ = (σ²).^2
    nm = 4.*ρ.^2.*σ⁴./((1-ρ).^6.*(1+ρ).^2)
    dn = σ⁴./(1-ρ).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.*ρ.^2.*σ⁴./(1-ρ).^8
    α₂ = sum(w.*nm)/sum(w.*dn)
    return α₁, α₂
end


## --> Newey-West Optimal bandwidth <---
growthrate(k::HAC) = 1/5
growthrate(k::BartlettKernel) = 1/3
lagtruncation(k::BartlettKernel) = 2/9
lagtruncation(k::ParzenKernel) = 4/25
lagtruncation(k::QuadraticSpectralKernel) = 2/25

bwnw(k::TruncatedKernel, s0, s1, s2) = error("Newey-West optimal bandwidth does not support TuncatedKernel")
bwnw(k::TukeyHanningKernel, s0, s1, s2) = error("Newey-West optimal bandwidth does not support TukeyHanningKernel")
bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447*((s1/s0)^2)^growthrate(k)
bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614*((s2/s0)^2)^growthrate(k)
bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221*((s2/s0)^2)^growthrate(k)

## --> Interface




function bwAndrews{T}(X::Matrix{T}, k::HAC, prewhite::Bool)
  isempty(k.weights) && (k.weights = ones(p))
  bwAndrews(X, k, k.weights, prewhite)
end

function bwAndrews{T}(X::Matrix{T}, k::HAC, w::Vector, prewhite::Bool)
    !prewhite || ((X, D) = pre_white(X))
    N, p  = size(X)
    a1, a2 = getalpha(X, :ar, w)
    return bw_andrews(k, a1, a2, N)
end


function bwAndrews(r::DataFrameRegressionModel, k::HAC, w::Array, prewhite::Bool)
    u = wrkresidwts(r.model.rr)
    X = ModelMatrix(r.model)
    z = X.*u
    p = size(z, 2)
    bwAndrews(z, k, w, prewhite)
end



function bwNeweyWest(r::DataFrameRegressionModel, k::HAC, w::Array, prewhite::Bool)
    u = wrkresidwts(r.model.rr)
    X = ModelMatrix(r.model)
    z = X.*u
    p = size(z, 2)
    bwNeweyWest(z, k, w, prewhite)
end

function bwNeweyWest{T}(X::Array{T, 2}, k::HAC, prewhite::Bool)
  isempty(k.weights) && (k.weights = ones(p))
  bwAndrews(X, k, k.weights, prewhite)
end

function getrates(X, k, prewhite)
  N, p = size(X)
  lrate = lagtruncation(k)
  adj = prewhite ? 3 : 4
  l = floor(Int, adj*(N/100)^lrate)
  n = ifelse(prewhite, N-1, N)::Int
  n, l
end

function bwNeweyWest{F<:Number}(X::Matrix, k::HAC, w::Vector{F}, prewhite::Bool)
    # N, p = size(X)
    # n = ifelse(prewhite, N-1, N)::Int
    # lrate = lagtruncation(k)
    # adj = prewhite ? 3 : 4
    # l = floor(Int, adj*(N/100)^lrate)
    n, l = getrates(X, k, prewhite)
    ## Prewhite if necessary
    !prewhite || ((X::Array{Float64,2}, D) = pre_white(X))
    gr = growthrate(k)
    N = n + ifelse(prewhite, 1, 0)
    bw_neweywest(k, X, w, l, gr, n, N)
  end

  function bw_neweywest(k, X, w, l, gr, n, N)
    #!prewhite || (n = (N - 1)::Int)
    xm = Array{Float64}(n)
    A_mul_B!(xm, X, w)

    ## Calculate truncated variance
    a = map(j -> dot(xm[1:n-j], xm[j+1:n])/n, 0:l)::Array{Float64, 1}
    aa = view(a, 2:l+1)
    a0 = a[1] + 2*sum(aa)
    a1 = 2*sum((1:l) .* aa)
    a2 = 2*sum((1:l).^2 .* aa)
    bwnw(k, a0, a1, a2)*N^gr
end


## -> Optimal bandwidth API

function stdregweights(r::DataFrameRegressionModel)
  nc = length(coef(r))::Int
  w = ones(nc)
  "(Intercept)" ∈ coefnames(r.mf) &&
  (w[find("(Intercept)" .== coefnames(r.mf))] = 0)
  w
end

optimal_bw(X::Matrix, k::HAC, t::NeweyWest, w::Array, prewhite::Bool) = bwNeweyWest(X, k, w, prewhite)
optimal_bw(X::Matrix, k::HAC, t::Andrews, w::Array, prewhite::Bool) = bwAndrews(X, k, w, prewhite)

optimal_bw(r::DataFrame, k::HAC, t::NeweyWest, w::Array, prewhite::Bool) = bwNeweyWest(r, k, w, prewhite)
optimal_bw(r::DataFrame, k::HAC, t::Andrews, w::Array, prewhite::Bool) = bwAndrews(r, k, w, prewhite)


optimalbw{K<:HAC, T}(t::Type{NeweyWest}, k::Type{K}, X::Matrix{T};
                      prewhite::Bool = false, weights = ones(size(X,2))) = bwNeweyWest(X, k(), weights, prewhite)

optimalbw{K<:HAC, T}(t::Type{Andrews}, k::Type{K}, X::Matrix{T};
                      prewhite::Bool = false, weights = ones(size(X,2))) = bwAndrews(X, k(), weights, prewhite)

optimalbw{K<:HAC}(t::Type{NeweyWest}, k::Type{K}, r::DataFrameRegressionModel;
                      prewhite::Bool = false, weights = stdregweights(r)) = bwNeweyWest(r, k(), weights, prewhite)

optimalbw{K<:HAC}(t::Type{Andrews}, k::Type{K}, r::DataFrameRegressionModel;
                      prewhite::Bool = false, weights = stdregweights(r)) = bwAndrews(r, k(), weights, prewhite)
