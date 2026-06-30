#=========
Value equality for estimators

Estimators are value objects: two built from the same specification compare equal
and hash equally, so they can serve as `Dict` keys or `unique` elements. Equality
reflects the *specification*, never transient fit-state that `aVar`/`optimalbw`
populate (HAC `bw`/`kw`/`wlock`, VARHAC `AICs`/`BICs`/orders); a fitted estimator
stays `==` its unfitted self.

`Base`'s default field-wise `==` only applies to `isbits` immutable structs;
anything holding a `Vector` (`CR*`, `DifferentOwnLags`, …) or a `mutable struct`
(`VARHAC`) otherwise falls back to `===`. The generic field-wise methods below
restore value semantics for every estimator and its component value types, and
`hash` is kept consistent with `==`. HAC and VARHAC override the generic method to
compare only specification fields.
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

# HAC kernels: the specification is the concrete kernel type and bandwidth type.
# For a fixed bandwidth the user-supplied value is part of the spec; for Andrews
# and Newey-West the `bw` vector is computed fit-state and is excluded.
_hac_spec(k::HAC) = ()
_hac_spec(k::HAC{Fixed}) = (first(k.bw),)

function Base.:(==)(a::HAC, b::HAC)
    typeof(a) === typeof(b) || return false
    return _hac_spec(a) == _hac_spec(b)
end
Base.hash(k::HAC, h::UInt) = hash(_hac_spec(k), hash(typeof(k), h))

# VARHAC: the specification is the lag selector and lag strategy; the mutable
# AIC/BIC tables and selected orders are fit-state and are excluded.
Base.:(==)(a::VARHAC, b::VARHAC) = a.selector == b.selector && a.strategy == b.strategy
Base.hash(k::VARHAC, h::UInt) = hash(k.strategy, hash(k.selector, hash(:VARHAC, h)))
