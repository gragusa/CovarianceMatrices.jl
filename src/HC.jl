struct HCCache{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector}
    q::F1
    X::F1
    x::F2
    v::V
    w::V
    η::V
    u::V
end

function HCCache(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    HCConfig(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n))
end
