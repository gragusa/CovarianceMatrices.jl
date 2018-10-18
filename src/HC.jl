struct HCCache{T1<:Real, F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector}
    q::F1
    X::F1
    x::F2
    v::V
    w::V
    η::V
    u::V
    chol::Cholesky{T1, F1}
end

function HCCache(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    HCCache(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n), cholesky(Matrix(one(T1)I, p, p)))
end
