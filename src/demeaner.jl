function demeaner(
        X::AbstractMatrix{T};
        means::Union{Nothing, AbstractArray} = nothing,
        dims = 1
) where {T <: Real}
    ## dims = 1 - calculate means for each columns
    ## dims = 2 - calculate mean for each row
    Z = if means === nothing
        X .- mean(X; dims = dims)
    else
        n, p = size(means)
        m, r = size(X)
        dims == 1 ?
        p == r || Base.throw(
            ArgumentError(
            "The `means` vector is of dimension ($n x $p). It should be of dimension ($p x $n)",
        ),
        ) :
        n == m || Base.throw(
            ArgumentError(
            "The `means` vector is of dimension ($n x $p). It should be of dimension ($p x $n)",
        ),
        )
        X .- means
    end
    return dims == 1 ? Z : collect(Z')
end
