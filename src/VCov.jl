module VCov


export vcovHAC

## Interface

## OLS.pp.chol contains the chol(X'X) matrix
##


##vcov(obj, ::Metho)

# abstract SmoothingKernel

# type BandWidth{T}
#   bw::T
# end

# type BartlettKernel
#   bw::BandWidth
#   kernel::Function
#   idxs::Indexes
# end

# function BartlettKernel(bw::BandWidth)
#   BartlettKernel(bw, )


# function vcovHAC{T}(g::AbstractMatrix{T}, sk::SmoothingKernel)
#   n, m = size(g)
#   l = int(floor(4*((n/100)^(2/9))))
#   vcovHAC(g, l)
# end

# function vcovHAC(g::Array{Float64, 2}, l::Int64)
#   n, m = size(g)
#   Q = g'g
#   for ℓ = 1:l
#     ω = 1-ℓ/(l+1)
#     for t = ℓ+1:n
#       Q += ω * (g[t, :]' * g[t-ℓ,:] + g[t-ℓ, :]' * g[t,:])
#     end
#   end
#   return Q/n
# end



# BartlettKernel() = BartlettKernel(opt_kernel, )


# function bartlett_kernel(x::Vector)
#   nx = length(x)
#   for j = 1:nx
#     axj = abs(x[j])
#     if(axj <=1)
#       x[j] = 1-axj
#     end
#   end
# end

# function bartlett_kernel(x::Number)
#   if(x <=1)
#     return  1-abs(x)
#   elseif
#     return 0.0
#   end
# end

# function kernelsmooth{T}(g::AbstractMatrix{T}, ker::SmoothingKernel)
#   nr, nc = dim(g)
#   M = Array{T, nc, nc}
#   for j = 1:nr

function vcovHAC(g::Array{Float64, 2})
  n, m = size(g)
  l = int(floor(4*((n/100)^(2/9))))
  vcovHAC(g, l)
end

function vcovHAC(g::Array{Float64, 2}, l::Int64)
  n, m = size(g)
  Q = g'g
  for ℓ = 1:l
    ω = 1-ℓ/(l+1)
    for t = ℓ+1:n
      Q += ω * (g[t, :]' * g[t-ℓ,:] + g[t-ℓ, :]' * g[t,:])
    end
  end
  return Q/n
end




end # module
