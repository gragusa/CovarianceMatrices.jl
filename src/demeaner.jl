function demeaner(X::AbstractMatrix{T}; means::Union{Nothing, AbstractArray}=nothing, dims=1) where {T<:Real}
    ## dims = 1 - calculate means for each columns
    ## dims = 2 - calculate mean for each row
    Z = if means === nothing
        X .- mean(X; dims=dims)
    else
        n, p = size(means)
        m, r = size(X)
        dims == 1 ? p == r || Base.throw(ArgumentError("The `means` vector is of dimension ($n x $p). It should be of dimension ($p x $n)")) : n == m || Base.throw(ArgumentError("The `means` vector is of dimension ($n x $p). It should be of dimension ($p x $n)"))
        X .- means
    end
    dims == 1 ? Z : collect(Z')
end

## TODO - Implement the following function
## Not working now because changed the way the clusterindicator is implemented
function demeaner(k::CR, X::AbstractMatrix{T}; dims=1, kwargs...) where T<:AbstractFloat
    ## dims = 1 - calculate means for each columns
    ## dims = 2 - calculate mean for each row
    ## Calculate partitions mean
    f = clusterindicator(k)
    Z = dims==1 ? copy(X) : collect(X')
    for j in clusterintervals(f)
        W = view(Z, j, :)
        W .= W .- mean(W, dims = 1)
    end
    return Z
end
