<<<<<<< HEAD
struct HCCache{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector} <: AbstractCache
=======
struct HCCache{T1<:Real, F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector}
>>>>>>> newcache
    q::F1
    X::F1
    x::F2
    v::V
    w::V
    η::V
    u::V
<<<<<<< HEAD
end

function HCCache(X::AbstractMatrix{T1}) where T1<:Real
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
=======
    chol::Cholesky{T1, F1}
end

function HCCache(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    HCCache(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n), cholesky(Matrix(one(T1)I, p, p)))
end
>>>>>>> newcache
