clusterindicator(x::CR) = x.cl
clusterintervals(f::CategoricalArray) = (searchsorted(f.refs, j) for j in unique(f.refs))
avarscaler(K::CR) = length(unique(clusterindicator(K)))

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

function avar(k::T, X) where T<:CR
    f = clusterindicator(k)
    if !issorted(f) 
        idx = sortperm(f)
        X = X[idx, :]
    end
    clustersum!(X, f[idx])
end

function clustersum!(X::Matrix{T}, cl) where T<:Real
    n, p = size(X)
    Shat = zeros(T, p, p)
    s = Array{T}(undef, p)

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