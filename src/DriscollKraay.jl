"""
DriscolKraay variance-covariance matrix estimator

Driscol and Kraay (1998) 
"""

function avar(k::T, X::Matrix{R}; kwargs...) where {T<:DriscollKraay,R<:Real}
    tis = k.tis
    iis = k.iis
    X2 = zeros(eltype(X), tis.ngroups, size(X, 2))
    idx = 0
    for j ∈ 1:size(X, 2)
        idx += 1
        @inbounds @simd for i ∈ 1:size(X, 1)
            X2[tis.groups[i], idx] += X[i, j]
        end
    end
    return a𝕍ar(k.K, X2; kwargs...)
end
