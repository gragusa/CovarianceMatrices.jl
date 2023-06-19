function avar(k::T, X; kwargs...) where T<:CR    
    f = clusterindicator(k)
    length(f) != size(X,1) && throw(ArgumentError("The length of the cluster indicator, $(length(f)), does not match the number of row of the matrix, $(size(X,1))"))
    #M = issorted(f) ? clustersum(X, f) : (i = sortperm(f); clustersum(X[i, :], f[i]))
    M = clustersum(parent(X), f)
    ## G/n^2 M
    G = length(levels(f))
    n = length(f) 
    (G.*M./n^2)
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
        map(A) do M
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
    _, p = size(X)
    for m in clusterintervals(cl)
        @inbounds fill!(s, zero(T))
        innerXiXi!(s, m, X)
        innerXiXj!(Shat, s)         
    end
    return LinearAlgebra.copytri!(Shat, 'U')
end

function innerXiXi!(s, m, X)
    @tturbo for j in eachindex(s)
        for i in eachindex(m)
            s[j] += X[m[i], j]
        end
    end
end

function innerXiXj!(Shat, s) 
    @inbounds for j in eachindex(s)
        @tturbo for i in 1:j
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
