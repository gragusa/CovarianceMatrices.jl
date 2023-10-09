Base.@propagate_inbounds function Λ(j::Integer, m::AbstractMatrix{F}) where F<:AbstractFloat
  T, p = size(m)
  L = similar(m, (p,1))
  fill!(L, zero(F))
  for t in 1:T
    w = cos((π*j*((t-0.5)/T)))
    z = view(m, t, :)
    L .= L .+ w.*z
  end
  return L *= sqrt(2/T)
end


function ewc_weights(ν, T)
  fvec = collect(pi.*(1:ν))
  tvec = collect((0.5:T)./T)
  ω = (sqrt(2)/sqrt(T))*cos(tvec*fvec')
  return ω
end


function avar(k::EWC, X::Matrix{F}) where {F<:AbstractFloat}
  B = k.B
  T, p = size(X)
  Ω = similar(X, (p,p))
  fill!(Ω, zero(F))
  for j in 1:B
    L = Λ(j, X)
    @. Ω += L*L'
  end
    @. Ω /= B
end


