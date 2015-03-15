function lag!{T}(Yl::Array{T, 2}, Y::Array{T, 1}, p::Int64)
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

function olsvar{T}(y::Array{T, 2})
    # Input : data y, lag order 1
    # Output: coefficient estimate β, residual U
    N, K = size(y)
    Y = y[2:N,:]  # LHS variable
    X = y[1:N-1,:]
    A = X\Y    # OLS estimator
    U = Y-X*A  # estimated residuals
    return U, A'
end

function ar{T}(Y::Array{T, 2}, lag::Int64)
	   N, p = size(Y)
	   Yl = Array(T, N-lag, lag+1)
	   ρ  = Array(T, p, lag)
	   σ² = Array(T, p)
	   for j = 1:p
		      lag!(Yl, Y[:,j], lag)
		      Yt          = Y[lag+1:end,j]
		      θ           = (Yl\Yt)
		      ρ[j, 1:lag] = θ[1:lag]
		      ε           = Yt - Yl*θ
		      σ²[j]       = (ε'ε/N)[1]
	   end
	   return ρ, σ²
end

function arma{T}(Y::Array{T,2})
    ## Estimate an ARMA(1,1) for each column of Y
end

ar{T}(Y::Array{T, 2}) = ar(Y, 1)

d_bw_andrews = @compat Dict(:TruncatedKernel         => :(0.6611*(a2*N)^(0.2)),
                            :BartlettKernel          => :(1.1447*(a1*N)^(1/3)),
                            :ParzenKernel            => :(2.6614*(a2*N)^(0.2)),
                            :QuadraticSpectralKernel => :(1.3221*(a2*N)^(0.2)))


for tty in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :QuadraticSpectralKernel]
    @eval  $:(bw_andrews)(k::($tty), a1, a2, N) = $(d_bw_andrews[tty])
end

function pre_white(X::AbstractMatrix)
    X, D = olsvar(X)
    (X, inv(eye(size(X, 2))-D))
end

function getalpha(X::AbstractMatrix, approx::Symbol)
    ## @assert approx == :ar ## || approx == :arma
    ## if approx == :ar
        ρ, σ² = ar(X)
        σ⁴    = (σ²).^2
        nm    = 4.*ρ.^2.*σ⁴./((1-ρ).^6.*(1+ρ).^2)
        dn    = σ⁴./(1-ρ).^4
        α₁    = sum(nm)/sum(dn)
        nm    = 4.*ρ.^2.*σ⁴./(1-ρ).^8
        α₂    = sum(nm)/sum(dn)
    ## elseif approx == :arma  [TODO]
    ## end
    return α₁, α₂
end

function optimalbw_ar_one(X::AbstractMatrix, k::TruncatedKernel)
    a1, a2 = getalpha(X, :ar)
    T, p   = size(X)
    return .6611*(a2*T)^(1/5)
end

function optimalbw_ar_one(X::AbstractMatrix, k::BartlettKernel)
    a1, a2 = getalpha(X, :ar)
    T, p   = size(X)
    return 1.1447*(a1*T)^(1/3)
end

function optimalbw_ar_one(X::AbstractMatrix, k::ParzenKernel)
    a1, a2 = getalpha(X, :ar)
    T, p   = size(X)
    return 2.6614*(a2*T)^(1/5)
end

function optimalbw_ar_one(X::AbstractMatrix, k::QuadraticSpectralKernel)
    a1, a2 = getalpha(X, :ar)
    T, p   = size(X)
    return 1.3221*(a2*T)^(1/5)
end

function bwAndrews{T}(X::Array{T, 2}, k::HAC; prewhite::Bool = false, approx::Symbol = :ar)
    N, p  = size(X)
    !prewhite || ((X, D) = pre_white(X))
    a1, a2 = getalpha(X, approx)
    return bw_andrews(k, a1, a2, N)
end
