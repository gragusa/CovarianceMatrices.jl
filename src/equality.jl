#=========
Value equality for estimators

Estimators are value objects: two built from the same specification compare equal
and hash equally, so they can serve as `Dict` keys or `unique` elements. Equality
reflects the specification, which is all an estimator carries: quantities selected
from data live in the returned `CovarianceMatrix`, not in the estimator.

`Base`'s default field-wise `==` only applies to `isbits` immutable structs;
anything holding a `Vector` (`CR*`, `DifferentOwnLags`, …) otherwise falls back to
`===`. The generic field-wise methods below restore value semantics for every
estimator and its component value types, and `hash` is kept consistent with `==`.
=========#

# Recursive field-wise value comparison and hashing. `==` already recurses into
# the per-field `==`, so component value types (Clustering, lag strategies)
# participate once they have their own method.
function _fieldwise_equal(a, b)
    typeof(a) === typeof(b) || return false
    for f in 1:nfields(a)
        getfield(a, f) == getfield(b, f) || return false
    end
    return true
end

function _fieldwise_hash(x, h::UInt)
    h = hash(typeof(x), h)
    for f in 1:nfields(x)
        h = hash(getfield(x, f), h)
    end
    return h
end

# Estimators and the bare cache-wrapper that does not subtype the estimator root.
for T in (:AbstractAsymptoticVarianceEstimator, :CachedCRModel)
    @eval Base.:(==)(a::$T, b::$T) = _fieldwise_equal(a, b)
    @eval Base.hash(x::$T, h::UInt) = _fieldwise_hash(x, h)
end

# Component value types carried inside estimators.
for T in (:Clustering, :BandwidthType, :LagSelector, :LagStrategy)
    @eval Base.:(==)(a::$T, b::$T) = _fieldwise_equal(a, b)
    @eval Base.hash(x::$T, h::UInt) = _fieldwise_hash(x, h)
end

