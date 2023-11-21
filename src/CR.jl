function clusterize(X::Matrix, g::GroupedArray)
  X2 = zeros(eltype(X), g.ngroups, size(X, 2))
  idx = 0  
  for j in 1:size(X, 2)
          idx += 1
          @inbounds @simd for i in 1:size(X, 1)
              X2[g.groups[i], idx] += X[i, j]
          end
      end
  return Symmetric(X2' * X2)
end

function avar(k::T, X::Union{Matrix{F},Vector{F}}; kwargs...) where {T<:CR0, F<:AbstractFloat}    
  f = k.g
  S = zeros(eltype(X), (size(X, 2), size(X,2)))
  for c in combinations(1:length(f))    
    if length(c) == 1
        g = GroupedArray(f[c[1]])
    else
        g = GroupedArray((f[i] for i in c)..., sort = nothing)
    end    
    S += (-1)^(length(c) - 1) * clusterize(parent(X), g)
  end
  return Symmetric(S)
end

##function _avar(k::T, X::Union{Matrix{F},Vector{F}}, f::C; kwargs...) where {T<:CR, F<:AbstractFloat, C<:Tuple}
##  f1 = f[1]
##  length(f1) != size(X,1) && throw(ArgumentError("The length of the cluster indicators is $(length(f1)) while it should be $(size(X,1))"))
##  M1 = clustersum(parent(X), f1)
##  if length(f) == 1
##    return M1
##  end
##  f2 = f[2]
##  length(f2) != size(X,1) && throw(ArgumentError("The length of the cluster indicators is $(length(f2)) while it should be $(size(X,1))"))
##  M2 = clustersum(parent(X), f2)
##  f0 = CategoricalArray(GroupedArray(f1, f2).groups)
##  M0 = clustersum(parent(X), f0)
##  return M1+M2-M0
##end

clusterindicator(x::CR) = x.cl

function clusterintervals(f::CategoricalArray)
  if issorted(f)
      (searchsorted(f.refs, j) for j in unique(f.refs))
  else
      (findall(f.refs.==j) for j in unique(f.refs))
  end
end


avarscaler(K::CR, X)  = length(unique(clusterindicator(K)))
avarscaler(K::HR, X)  = size(X, 1)

function sortrowby(A, by) 
    if !issorted(by) 
        sortedperm = sortperm(by); 
        mapreduce(hcat, eachcol(A)) do M
            let sortedperm = sortedperm 
                M[sortedperm, :]
            end
        end, by[sortedperm] 
    else 
        return A, by
    end
end

function clustersum(X::Vector{T}, cl) where T<:Real
    Shat = fill!(similar(X, (1, 1)), zero(T))
    s = Vector{T}(undef, size(Shat, 1))
    clustersum!(Shat, s, X[:,:], cl)
    vec(Shat)
end

function clustersum(X::Matrix{T}, cl) where T<:Real 
    _, p = size(X)
    Shat = fill!(similar(X, (p, p)), zero(T))
    s = Vector{T}(undef, size(Shat, 1))
    clustersum!(Shat, s, X, cl)
end

function clustersum!(Shat::Matrix{T}, s::Vector{T}, X::Matrix{T}, cl) where T<:Real
    for m in clusterintervals(cl)
        @inbounds fill!(s, zero(T))
        innerXiXi!(s, m, X)
        innerXiXj!(Shat, s)
    end
    return LinearAlgebra.copytri!(Shat, 'U')
end

function innerXiXi!(s, m, X)
    @inbounds @fastmath for j in eachindex(s)
        for i in eachindex(m)
            s[j] += X[m[i], j]
        end
    end
end

function innerXiXj!(Shat, s)
    @inbounds @fastmath for j in eachindex(s)
         for i in 1:j
            Shat[i, j] += s[i]*s[j]
        end
    end
end


function clustersum_slow(X::Matrix{T}, cl) where T<:Real
    _, p = size(X)
    Shat = fill!(similar(X, (p, p)), zero(T))
    s = Vector{T}(undef, size(Shat, 1))
    clustersum!(Shat, s, X, cl)
end

function clustersum_slow!(Shat::Matrix{T}, s::Vector{T}, X::Matrix{T}, cl) where T<:Real
    _, p = size(X)
    for m in clusterintervals(cl)
        fill!(s, zero(T))
        for j in 1:p
            for i in m
                s[j] += X[i, j]
            end
        end
        for j in 1:p
            for i in 1:j
                Shat[i, j] += s[i]*s[j]
            end
        end
    end
    return LinearAlgebra.copytri!(Shat, 'U')
end
