module SparseArraysExt

using CovarianceMatrices
using CovarianceMatrices: Clustering
using LinearAlgebra: LinearAlgebra, Symmetric, qr!
using SparseArrays: SparseArrays, AbstractSparseMatrix, nonzeros, nzrange, rowvals

function CovarianceMatrices.clusterize(X::AbstractSparseMatrix, g::Clustering)
    m, n = size(X)
    X2 = zeros(eltype(X), g.ngroups, n)

    rows = rowvals(X)
    vals = nonzeros(X)

    @inbounds for j in 1:n
        for i in nzrange(X, j)
            row = rows[i]
            val = vals[i]
            X2[g.groups[row], j] += val
        end
    end

    return Symmetric(X2' * X2)
end

Base.@propagate_inbounds function CovarianceMatrices.fit_var(
        A::AbstractSparseMatrix{T},
) where {T}
    P = parent(A)
    li = lastindex(P, 1)
    Y = P[2:li, :]
    X = P[1:(li - 1), :]
    B = qr!(X'X) \ Matrix(X'Y)
    E = Y - X * B
    return E, B
end

end
