avar(k::T, X::M; kwargs...) where {T <: HR, M <: AbstractMatrix} = X'X

# `Uncorrelated` shares the heteroskedasticity-robust "meat":
#   Ω̂ = ∑ₜ gₜ gₜ' = X'X.
# On a fitted model this produces White's HC0 sandwich (no leverage/DOF adjustment);
# see `residual_adjustment(::Uncorrelated, ::RegressionModel)` in
# `regression_model_estimators.jl`.
avar(k::Uncorrelated, X::M; kwargs...) where {M <: AbstractMatrix} = X'X
