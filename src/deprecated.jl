#=========
Deprecated call forms

Fit results moved from the estimator to the `CovarianceMatrix` that `aVar` and
`vcov` return. The shims below keep the old reads working with a warning; they are
removed in a later release.
=========#

# `kw` and `wlock` are not fields of any estimator. `bw` is a field of a
# fixed-bandwidth kernel and is reached through `getfield`; the reads below return
# the empty values the removed boxes held.
function Base.getproperty(k::HAC, s::Symbol)
    if s === :kw
        Base.depwarn(
            "`k.kw` is deprecated: kernel weights are a property of an estimate, not of the estimator. Use `kernelweights(aVar(k, X))`.",
            :getproperty)
        return WFLOAT[]
    elseif s === :wlock
        Base.depwarn(
            "`k.wlock` is deprecated: estimators no longer carry mutable fit-state, so there is nothing to lock.",
            :getproperty)
        return [false]
    end
    return getfield(k, s)
end

for f in (:AICs, :BICs, :order_aic, :order_bic, :order)
    @eval function $f(k::VARHAC)
        Base.depwarn(
            string("`", $(QuoteNode(f)),
                "(k::VARHAC)` is deprecated: lag selection is a property of an estimate, not of the estimator. Use `",
                $(QuoteNode(f)), "(aVar(k, X))`."),
            $(QuoteNode(f)))
        return nothing
    end
end
