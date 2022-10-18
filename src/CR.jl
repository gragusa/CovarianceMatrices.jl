function avar(k::T, X; kwargs)
    Z = copy(X)
    avar!(k, X; kwargs...)
end

avar!(k::T, X; kwargs...)
    ci = clusterindicators(k)
    ## Sort in place if needed
    