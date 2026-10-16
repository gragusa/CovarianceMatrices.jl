"""
Driscoll-Kraay variance-covariance matrix estimator

Driscoll and Kraay (1998)
"""

# The HAC kernel is applied to the series of period sums without further demeaning or
# scaling: `aVar` demeans the observation-level moments (so the period sums have mean
# zero) and divides by the number of observations, as for every other estimator.
# The inner kernel's selected quantities (bandwidth, kernel weights) are reported
# as the Driscoll-Kraay estimate's own.
function avar_with_info(k::DriscollKraay, X; weights = nothing, prewhite = false)
    k.correction === :HC0 || throw(ArgumentError(
        "the Driscoll-Kraay correction $(repr(k.correction)) needs the number of " *
        "coefficients, which a moment matrix does not carry; use `vcov(k, model)` " *
        "with a `RegressionModel`, or `correction = :HC0`"))
    return avar_with_info(k.K, _periodsums(k, X); prewhite, weights)
end

avar(k::DriscollKraay, X; kwargs...) = first(avar_with_info(k, X; kwargs...))

function _periodsums(k::DriscollKraay, X::AbstractMatrix)
    Base.require_one_based_indexing(X)
    tis = k.tis
    length(tis.groups) == size(X, 1) || throw(DimensionMismatch(
        "DriscollKraay: `tis` has $(length(tis.groups)) observations but the moment matrix has $(size(X, 1)) rows"))
    X2 = zeros(eltype(X), tis.ngroups, size(X, 2))
    for j in axes(X, 2), i in axes(X, 1)

        X2[tis.groups[i], j] += X[i, j]
    end
    return X2
end

_prewhitens(k::DriscollKraay) = _prewhitens(k.K)

# Bandwidth selection weights come from the model matrix, as for the inner kernel
# applied directly: the intercept column gets weight 0.
bandwidthweights(k::DriscollKraay, X::AbstractMatrix) = bandwidthweights(k.K, X)

"""
    timekernel(k::DriscollKraay) -> HAC

Return the HAC kernel that `k` applies to the series of period sums.
"""
timekernel(k::DriscollKraay) = k.K

"""
    timeindex(k::DriscollKraay) -> Clustering

Return the time period of each observation as a [`Clustering`](@ref
CovarianceMatrices.Clustering) whose group codes follow the chronological order of
the periods.
"""
timeindex(k::DriscollKraay) = k.tis

"""
    unitindex(k::DriscollKraay) -> Clustering

Return the unit of each observation as a [`Clustering`](@ref
CovarianceMatrices.Clustering).

`DriscollKraay(timekernel(k), timeindex(k), unitindex(k))` rebuilds `k` without its
finite-sample correction, which is the estimator `aVar` accepts on a moment matrix.
"""
unitindex(k::DriscollKraay) = k.iis
