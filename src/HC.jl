

function HCCache(X::AbstractMatrix{T1}; kwargs...) where T1<:Real
    n, p = size(X)
    HCCache(similar(X), X, Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n))
end

function emptyHCCache()
    V = Array{WFLOAT, 1}(undef, 0)
    M = Array{WFLOAT, 2}(undef, 0, 0)
    HCCache(M, M, M, V, V, V, V)
end
