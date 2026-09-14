#=========
CovarianceMatrix — an estimate together with the specification that produced it
=========#
"""
    CovarianceMatrix{T,M,E,I} <: AbstractMatrix{T}

An estimated covariance matrix together with the estimator that produced it and
the quantities selected during estimation.

`aVar` and `vcov` return this type. It behaves as an `AbstractMatrix`: indexing,
`size`, matrix arithmetic, `inv`, `\\`, `diag` and factorizations all work
through the `AbstractMatrix` fallbacks. Operations that do not preserve the
covariance interpretation — `V * V`, `V + V` — return a plain matrix.

The stored matrix is symmetrized at construction: a sandwich estimate is
symmetric only to floating-point tolerance, and the symmetrization is applied
explicitly rather than hidden behind a `Symmetric` view that would discard one
triangle.

# Fields
- `V`: the estimate.
- `estimator`: the estimator specification, an immutable value.
- `info`: a `NamedTuple` of quantities selected during estimation — `bandwidth`
  and `kernelweights` for HAC kernels, lag orders and information criteria for
  `VARHAC`, empty for estimators that select nothing.

# Accessors
[`estimator`](@ref), [`bandwidth`](@ref), [`kernelweights`](@ref),
[`information`](@ref), and `parent` for the bare matrix.

# Examples
```julia
V = aVar(Bartlett{Andrews}(), X)
bandwidth(V)        # the bandwidth this estimate was computed with
estimator(V)        # Bartlett{Andrews}()
parent(V)           # the underlying Matrix
V \\ b               # works through the AbstractMatrix fallbacks
```
"""
struct CovarianceMatrix{T, M <: AbstractMatrix{T}, E, I <: NamedTuple} <:
       AbstractMatrix{T}
    V::M
    estimator::E
    info::I

    # Symmetrize on the way in: a sandwich estimate is symmetric only to
    # floating-point tolerance. `NaN` entries, which mark non-estimable
    # parameters, propagate through unchanged.
    function CovarianceMatrix{T, M, E, I}(V, estimator, info) where {T, M, E, I}
        return new{T, M, E, I}(V, estimator, info)
    end
end

Base.size(V::CovarianceMatrix) = size(V.V)
Base.getindex(V::CovarianceMatrix, i::Int) = getindex(V.V, i)
Base.getindex(V::CovarianceMatrix, I::Vararg{Int, N}) where {N} = getindex(V.V, I...)
Base.IndexStyle(::Type{<:CovarianceMatrix{T, M}}) where {T, M} = IndexStyle(M)
Base.parent(V::CovarianceMatrix) = V.V
Base.axes(V::CovarianceMatrix) = axes(V.V)
Base.similar(V::CovarianceMatrix, ::Type{S}, dims::Dims) where {S} = similar(V.V, S, dims)

"""
    estimator(V::CovarianceMatrix)

Return the estimator specification that produced `V`.
"""
estimator(V::CovarianceMatrix) = V.estimator

"""
    information(V::CovarianceMatrix)

Return the `NamedTuple` of quantities selected during the estimation of `V`.
"""
information(V::CovarianceMatrix) = V.info

"""
    bandwidth(V::CovarianceMatrix)

Return the bandwidth used to compute `V`, or `nothing` if the estimator does not
select one.
"""
bandwidth(V::CovarianceMatrix) = get(V.info, :bandwidth, nothing)

"""
    kernelweights(V::CovarianceMatrix)

Return the per-column kernel weights used in the bandwidth selection for `V`, or
`nothing` if the estimator does not use them.
"""
kernelweights(V::CovarianceMatrix) = get(V.info, :kernelweights, nothing)

# The estimate is symmetric only to floating-point tolerance. Symmetrize on the
# way in so the stored matrix is exactly symmetric; `NaN` entries, which mark
# non-estimable parameters, propagate through unchanged.
function _symmetrize(V::AbstractMatrix)
    size(V, 1) == size(V, 2) || return V
    return (V .+ transpose(V)) ./ 2
end
_symmetrize(V::Symmetric) = V

function CovarianceMatrix(V::AbstractMatrix, estimator, info::NamedTuple = NamedTuple())
    Vs = _symmetrize(V)
    return CovarianceMatrix{eltype(Vs), typeof(Vs), typeof(estimator), typeof(info)}(
        Vs, estimator, info)
end

function Base.show(io::IO, ::MIME"text/plain", V::CovarianceMatrix)
    println(io, size(V, 1), "×", size(V, 2), " CovarianceMatrix from ", V.estimator, ":")
    return Base.print_matrix(io, V.V)
end
