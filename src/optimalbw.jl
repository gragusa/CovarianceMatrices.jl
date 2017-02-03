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

function olsvar{T}(y::AbstractArray{T, 2})
    # Input : data y, lag order 1
    # Output: coefficient estimate β, residual U
    N, K = size(y)
    Y = y[2:N,:]  # LHS variable
    X = y[1:N-1,:]
    A = X\Y    # OLS estimator
    U = Y-X*A  # estimated residuals
    return U, A'
end

function ar{T}(Y::AbstractArray{T, 2}, lag::Int64)
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

function arma{T}(Y::Array{T,2})
    ## Estimate an ARMA(1,1) for each column of Y
end

ar{T}(Y::AbstractArray{T, 2}) = ar(Y, 1)

d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*N)^(0.2)),
                    :BartlettKernel          => :(1.1447*(a1*N)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*N)^(0.2)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*N)^(0.2)))

for tty in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :QuadraticSpectralKernel]
    @eval $:(bw_andrews)(k::($tty), a1, a2, N) = $(d_bw_andrews[tty])
end

function pre_white(X::AbstractMatrix)
    X, D = olsvar(X)
    (X, inv(I-D))
end

function getalpha(X::AbstractMatrix, approx::Symbol, w::Vector)
    ## @assert approx == :ar ## || approx == :arma
    ## if approx == :ar
    ρ, σ² = ar(X)
    σ⁴ = (σ²).^2
    nm = 4.*ρ.^2.*σ⁴./((1-ρ).^6.*(1+ρ).^2)
    dn = σ⁴./(1-ρ).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.*ρ.^2.*σ⁴./(1-ρ).^8
    α₂ = sum(w.*nm)/sum(w.*dn)
    ## elseif approx == :arma  [TODO]
    ## end
    return α₁, α₂
end

function optimalbw_ar_one(X::AbstractMatrix, k::TruncatedKernel)
    T, p = size(X)
    a1, a2 = getalpha(X, :ar, ones(p))
    return .6611*(a2*T)^(1/5)
end

function optimalbw_ar_one(X::AbstractMatrix, k::BartlettKernel)
    T, p = size(X)
    a1, a2 = getalpha(X, :ar, ones(p))
    return 1.1447*(a1*T)^(1/3)
end

function optimalbw_ar_one(X::AbstractMatrix, k::ParzenKernel)
    T, p = size(X)
    a1, a2 = getalpha(X, :ar, ones(p))
    return 2.6614*(a2*T)^(1/5)
end

function optimalbw_ar_one(X::AbstractMatrix, k::QuadraticSpectralKernel)
    T, p = size(X)
    a1, a2 = getalpha(X, :ar, ones(p))
    return 1.3221*(a2*T)^(1/5)
end

function bwAndrews{T}(X::Array{T, 2}, k::HAC, prewhite::Bool)
    !prewhite || ((X, D) = pre_white(X))
    N, p  = size(X)
    isempty(k.weights) && (k.weights = ones(p))
    a1, a2 = getalpha(X, :ar, k.weights)
    return bw_andrews(k, a1, a2, N)
end

function bwAndrews{T}(X::Array{T, 2}, k::HAC, w::Vector, prewhite::Bool)
    !prewhite || ((X, D) = pre_white(X))
    N, p  = size(X)
    a1, a2 = getalpha(X, :ar, w)
    return bw_andrews(k, a1, a2, N)
end

function bwAndrews(r::DataFrameRegressionModel, k::HAC; prewhite::Bool = false)
    u = wrkresidwts(r.model.rr)
    X = ModelMatrix(r.model)
    z = X.*u
    p = size(z, 2)
    w = ones(p)
    "(Intercept)" ∈ coefnames(r.mf) &&
    (w[find("(Intercept)" .== coefnames(r.mf))] = 0)
    bwAndrews(z, k, w, prewhite)
end

growthrate(k::HAC) = 1/5
growthrate(k::BartlettKernel) = 1/3
lagtruncation(k::BartlettKernel) = 2/9
lagtruncation(k::ParzenKernel) = 4/25
lagtruncation(k::QuadraticSpectralKernel) = 2/25

bwnw(k::TruncatedKernel, s0, s1, s2) = error("truncatd kernel not supported")
bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447*((s1/s0)^2)^growthrate(k)
bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614*((s2/s0)^2)^growthrate(k)
bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221*((s2/s0)^2)^growthrate(k)


function bwNeweyWest{T}(X::Array{T, 2}, k::HAC; prewhite::Bool = false)
    N, p = size(X)
    lrate = lagtruncation(k)
    adj = prewhite ? 3 : 4
    l = floor(Int, adj*(N/100)^lrate)

    ## Prewhite if necessary
    !prewhite || ((X, D) = pre_white(X))

    ## Calculate truncated variance
    a0 = CovarianceMatrices.Γ(X, 0)
    a1 = zeros(a0)
    a2 = zeros(a0)
    for j in 1:l
        γ = 2.*CovarianceMatrices.Γ(X, j)
        a0 .= a0 + γ
        a1 .= a1 + j*γ
        a2 .= a2 + j^2*γ
    end
    bwnw(k, s0, s1, s2)*N^growthrate(k)
end
