"""
DriscolKraay variance-covariance matrix estimator

Driscol and Kraay (1998) 
"""

# The inner kernel's selected quantities (bandwidth, kernel weights) are reported
# as the Driscoll-Kraay estimate's own.
function avar_with_info(k::DriscollKraay, X; weights = nothing, kwargs...)
    V = _dk_inner(k, X; kwargs...)
    return parent(V), information(V)
end

avar(k::DriscollKraay, X; kwargs...) = parent(_dk_inner(k, X; kwargs...))

function _dk_inner(k::T, X::Matrix{R}; kwargs...) where {T <: DriscollKraay, R <: Real}
    tis = k.tis
    iis = k.iis
    X2 = zeros(eltype(X), tis.ngroups, size(X, 2))
    idx = 0
    for j in 1:size(X, 2)
        idx += 1
        @inbounds @simd for i in 1:size(X, 1)
            X2[tis.groups[i], idx] += X[i, j]
        end
    end
    return aVar(k.K, X2; kwargs...)
end
