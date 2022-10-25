function avar(k::T, X) where T<:CR
    f = clusterindicator(k)
    issorted(f) ? clustersum(X, f) : (i = sortperm(f); clustersum(X[i], f[i]))
end

clusterindicator(x::CR) = x.cl
clusterintervals(f::CategoricalArray) = (searchsorted(f.refs, j) for j in unique(f.refs))
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

function clustersum(X::Matrix{T}, cl) where T<:Real 
    _, p = size(X)
    Shat = fill!(similar(X, (p, p)), zero(T))
    s = Vector{T}(undef, size(Shat, 1))
    clustersum!(Shat, s, X, cl)
end

function clustersum!(Shat::Matrix{T}, s::Array{T}, X::Matrix{T}, cl) where T<:Real
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