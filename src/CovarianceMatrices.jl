"""
A package for dealing with the estimation of (co)variance matrices.

    1. Define CovarianceEstimator(s) which are robust to heteroskedasticity, time-dependence, amd spatial dependence (cluster)
    2. Optionally, interface with GLM.jl to provide estimates of coefficients' standard errors which are robust

These use cases utilize different parts of this package, make sure you read the documentation.
"""

module CovarianceMatrices

using Combinatorics
using GroupedArrays
using LinearAlgebra
using Statistics
using StatsBase
using NaNStatistics
using Base.Threads

include("types.jl")
include("HAC.jl")
include("CR.jl")
include("HR.jl")
include("DriscollKraay.jl")
include("demeaner.jl")
include("EWC.jl")
#include("api.jl")
include("smoothing.jl")
include("VARHAC.jl")
include("aVar.jl")

# New unified API
include("model_interface.jl")
include("variance_forms.jl")
include("stable_computation.jl")
include("api.jl")
export Andrews,
       NeweyWest,
       Bartlett,
       Parzen,
       QuadraticSpectral,
       Truncated,
       TukeyHanning,
       CR0,
       CR1,
       CR2,
       CR3,
       HR,
       HR0,
       HR1,
       HR2,
       HR3,
       HR4,
       HR4m,
       HR5,
       HR,
       HC0,
       HC1,
       HC2,
       HC3,
       HC4,
       HC4m,
       HC5,
       HAC,
       EWC,
       Smoothed,
       aVar,
       a𝕍ar,
       optimalbw,
 #      bread,
 #      momentmatrix,
 #      residualadjustment,
       vcov,
       stderror,
       BartlettSmoother,
       TruncatedSmoother,
       # New API exports
       Information,
       Misspecified,
       VarianceForm,
       MLikeModel,
       GMMLikeModel,
       score,
       objective_hessian,
       weight_matrix
    end
