Base.String(::Type{T}) where T<:TruncatedKernel = "Truncated"
Base.String(::Type{T}) where T<:ParzenKernel = "Parzen"
Base.String(::Type{T}) where T<:TukeyHanningKernel = "Tukey-Hanning"
Base.String(::Type{T}) where T<:BartlettKernel = "Bartlett"
Base.String(::Type{T}) where T<:QuadraticSpectralKernel = "Quadratic Spectral"

Base.size(x::CovarianceMatrix) = size(x.V)
Base.getindex(x::CovarianceMatrix, row, col) = getindex(x.V, row, col)
Base.eltype(x::CovarianceMatrix) = eltype(x.V)

LinearAlgebra.inv(x::CovarianceMatrix) = inv(x.F)
function LinearAlgebra.pinv(x::CovarianceMatrix{T}; kwargs...) where T<:SVD
    F = invfact(x)
    F'*F
end

invfact(x::CovarianceMatrix{T1}, lower::Bool = true) where T1<:Cholesky = !lower ? inv(x.F.U) : inv(x.F.L)

function invfact(x::CovarianceMatrix{T1}; regularize::Bool = true, atol::Real = 0.0, rtol::Real = (eps(real(float(one(eltype(x)))))*min(size(x)...))*iszero(atol)) where T1<:SVD
    Sr = similar(x.F.S)
    S = x.F.S
    Vt = x.F.Vt
    U = x.F.U
    if regularize
        maxabs = abs.(first(S))
        tol = max(rtol*maxabs, atol)
        @inbounds for j in eachindex(S)
            if S[j] <= tol
                Sr[j] = zero(eltype(x))
            else
                Sr[j] = 1/sqrt(S[j])
            end
        end
    end
    Diagonal(Sr)*Vt
end

function quadinv(g::AbstractMatrix, x::CovarianceMatrix)
    @assert size(x, 1) == size(g, 2)
    B = invfact(x)
    gw = B*Transpose(g)
    dot(gw, gw)
end

function quadinv(g::AbstractVector, x::CovarianceMatrix)
    @assert length(g) == size(x, 1)
    B = invfact(x)
    gw = B*g
    dot(gw, gw)
end

LinearAlgebra.eigmax(x::CovarianceMatrix{T1}) where {T1<:SVD} = first(x.F.S)
LinearAlgebra.eigmin(x::CovarianceMatrix{T1}) where {T1<:SVD} = last(x.F.S)

LinearAlgebra.eigmax(x::CovarianceMatrix{T1}) where {T1<:Cholesky} = eigmax(Matrix(x))
LinearAlgebra.eigmin(x::CovarianceMatrix{T1}) where {T1<:Cholesky} = eigmin(Matrix(x))

LinearAlgebra.Symmetric(x::CovarianceMatrix) = LinearAlgebra.Symmetric(x.V)

function LinearAlgebra.logdet(x::CovarianceMatrix{T1}) where {T1<:Cholesky}
    a = zero(eltype(x))
    @simd for j in 1:size(x, 1)
        @inbounds a += log.(x.F.U[j,j])
    end
    2*a
end

LinearAlgebra.logdet(x::CovarianceMatrix{T1}) where {T1<:SVD} = sum(log.(x.F.S))
