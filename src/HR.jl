avar(k::T, X::M; kwargs...) where {T <: HR, M <: AbstractMatrix} = X'X

# On a matrix of moments, `Uncorrelated` has the same meat as `HR0`:
#   Ω̂ = ∑ₜ gₜ gₜ' = X'X.
# On a fitted model it is the iid variance instead; see `_iid_vcov` in
# `regression_model_estimators.jl`.
avar(k::Uncorrelated, X::M; kwargs...) where {M <: AbstractMatrix} = X'X
