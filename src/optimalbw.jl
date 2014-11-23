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
		ε         = Yt - Yl*θ
		σ²[j] = (ε'ε/N)[1]
	end
	return ρ, σ²
end 


function getalpha(X::AbstractMatrix)
  ρ, σ² = ar(X, 1)  
  σ⁴ = (σ²).^2
  nm   = 4.*ρ.^2.*σ⁴./(1-ρ).^6.*(1+ρ).^2
  dn   = σ⁴./(1-ρ).^4  
  α₁ = sum(nm)/sum(dn)
  nm   = 4.*ρ.^2.*σ⁴./(1-ρ).^8
  α₂ = sum(nm)/sum(dn)
  return α₁, α₂	
end 

function optimalbw_ar_one(X::AbstractMatrix, k::TruncatedKernel)
	α₁, α₂ = getalpha(X)
	T, p = size(X)
	return .6611*(α₂*T)^(1/5)	  
end 

function optimalbw_ar_one(X::AbstractMatrix, k::BartlettKernel)
	α₁, α₂ = getalpha(X)
	T, p = size(X)
	return 1.1447*(α₁*T)^(1/3)	  
end 

function optimalbw_ar_one(X::AbstractMatrix, k::ParzenKernel)
	α₁, α₂ = getalpha(X)
	T, p = size(X)
	return 2.6614*(α₂*T)^(1/5)	  
end 

function optimalbw_ar_one(X::AbstractMatrix, k::QuadraticSpectralKernel)
	α₁, α₂ = getalpha(X)
	T, p = size(X)
	return 1.3221*(α₂*T)^(1/5)	  
end 

