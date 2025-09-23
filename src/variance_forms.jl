"""
Variance estimator forms for different model types and assumptions.

This module defines the abstract type hierarchy for variance estimation forms,
following the mathematical framework outlined in the package documentation.
"""

abstract type VarianceForm end

"""
    Information <: VarianceForm

Information matrix-based variance for correctly specified MLE models.
Uses V = H⁻¹ where H is the objective Hessian.

Only valid for exactly identified models (m = k).
"""
struct Information <: VarianceForm end

"""
    Robust <: VarianceForm

Robust sandwich variance for M-like estimators under misspecification.
Uses V = G⁻¹ΩG⁻ᵀ where G is the Jacobian and Ω is the long-run covariance.

Valid for exactly identified models (m = k).
"""
struct Robust <: VarianceForm end

"""
    CorrectlySpecified <: VarianceForm

Optimal GMM variance under correct moment specification.
Uses V = (G'Ω⁻¹G)⁻¹ where moments are correctly specified.

Valid for overidentified models (m > k).
"""
struct CorrectlySpecified <: VarianceForm end

"""
    Misspecified <: VarianceForm

Robust GMM variance allowing for moment misspecification.
Uses V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹ where W is a weighting matrix.

Valid for overidentified models (m > k).
If W is not provided, defaults to Ω⁻¹ (equivalent to CorrectlySpecified).
"""
struct Misspecified <: VarianceForm end

# Convenience type unions for dispatch
const MLikeForm = Union{Information,Robust}
const GMMLikeForm = Union{CorrectlySpecified,Misspecified}

"""
    auto_form(model) -> VarianceForm

Automatically select variance form based on model dimensions.
- If m == k (exactly identified): returns Robust() for safety
- If m > k (overidentified): returns CorrectlySpecified() for efficiency
"""
function auto_form(model)
    m = size(momentmatrix(model), 2)
    k = length(StatsBase.coef(model))

    if m == k
        return Robust()  # Safe default for M-like models
    elseif m > k
        return CorrectlySpecified()  # Preferred default for GMM
    else
        throw(ArgumentError("Invalid model: fewer moments (m=$m) than parameters (k=$k)"))
    end
end
