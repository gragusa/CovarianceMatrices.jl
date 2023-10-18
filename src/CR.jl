  abstract type Evaluation end
  struct Sequential  <: Evaluation end
  struct Threaded  <: Evaluation end

function avar(k::T, X::Union{Matrix{F},Vector{F}}, evaluation::Evaluation=Sequential(); kwargs...) where {T<:CR, F<:AbstractFloat}    
    f = clusterindicator(k)
    V = _avar(k, X, f, evaluation; kwargs...)
    return V
end

function _avar(k::T, X::Union{Matrix{F},Vector{F}}, f::C, evaluation::Evaluation=Sequential(); kwargs...) where {T<:CR, F<:AbstractFloat, C<:Tuple}
  f1 = f[1]
  length(f1) != size(X,1) && throw(ArgumentError("The length of the cluster indicators is $(length(f1)) while it should be $(size(X,1))"))
  M1 = clustersum(parent(X), f1, evaluation)
  if length(f) == 1
    return M1
  end
  f2 = f[2]
  length(f2) != size(X,1) && throw(ArgumentError("The length of the cluster indicators is $(length(f2)) while it should be $(size(X,1))"))
  M2 = clustersum(parent(X), f2, evaluation)
  f0 = f1 .& f2
  if sum(f0) == 0
    return M1+M2
  else
    M0 = clustersum(parent(X), f1 .& f2, evaluation)
    return M1+M2-2M0
  end
end

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
        A[sortedperm, :], by[sortedperm] 
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
