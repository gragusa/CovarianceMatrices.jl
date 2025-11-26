"""
A package for dealing with the estimation of (co)variance matrices.

    1. Define CovarianceEstimator(s) which are robust to heteroskedasticity, time-dependence, amd spatial dependence (cluster)
    2. Optionally, interface with GLM.jl to provide estimates of coefficients' standard errors which are robust

These use cases utilize different parts of this package, make sure you read the documentation.
"""

module CovarianceMatrices

using StatsAPI
using StatsAPI: vcov, stderror
using Combinatorics
using GroupedArrays
using LinearAlgebra
using Statistics
using StatsBase
using NaNStatistics
using Base.Threads
using BlockDiagonals
include("types.jl")
include("HAC.jl")
include("CR.jl")
include("HR.jl")
include("DriscollKraay.jl")
include("demeaner.jl")
include("EWC.jl")
include("smoothing.jl")
include("VARHAC.jl")
include("aVar.jl")
# New unified API
include("model_interface.jl")
include("stable_computation.jl")
include("api.jl")
# Method generic RegressionModel estimators
include("regression_model_estimators.jl")

export AbstractAsymptoticVarianceEstimator,
       Uncorrelated,
       Correlated,
       Andrews,
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
       vcov,
       stderror,
       DriscollKraay,
       SmoothedMoments,
       UniformSmoother,
       TriangularSmoother,
# VARHAC exports
       VARHAC,
       AICSelector,
       BICSelector,
       FixedSelector,
       SameLags,
       DifferentOwnLags,
       FixedLags,
       AutoLags,
# New API exports
       Information,
       Misspecified,
       VarianceForm,
       MLikeModel,
       GMMLikeModel,
       momentmatrix,
       cross_score,
       jacobian_momentfunction,
       hessian_objective,
       weight_matrix
end
