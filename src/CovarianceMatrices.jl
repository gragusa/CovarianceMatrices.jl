"""
A package for dealing with the estimation of (co)variance matrices.

    1. Define CovarianceEstimator(s) which are robust to heteroskedasticity, time-dependence, amd spatial dependence (cluster)
    2. Optionally, interface with GLM.jl to provide estimates of coefficients' standard errors which are robust

These use cases utilize different parts of this package, make sure you read the documentation.
"""

module CovarianceMatrices

using CategoricalArrays
using LinearAlgebra
using Requires: @require
using Statistics
using StatsBase
using StatsModels
using LoopVectorization
using SparseArrays
#using ForwardDiff
include("types.jl")
include("aVar.jl")
include("HAC.jl")
include("CR.jl")
include("demeaner.jl")

#include("HR.jl")
#include("burger.jl")
#include("glue_glm.jl")

#include("VARHAC.jl")
#include("lrvar.jl")
#include("smoothing.jl")
#include("CovarianceMatrix.jl")
#include("interface_api.jl")
# using GLM
# include("glm.jl")

# function __init__()
#     @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("glm.jl")
# end

export Andrews,
       Bartlett,
       CR0,
       CR1,
       CR2,
       CR3,
       Covariance,
       HR,
       HR0,
       HR1,
       HR2,
       HR3,
       HR4,
       HR4m,
       HR5,
       HAC,
       NeweyWest,
       Parzen,
       QuadraticSpectral,
       Truncated,
       TukeyHanning,
       Smoothed,
       aVar,
       a𝕍ar,
       optimalbw
    #    lrvar,
    #    lrvarmatrix,
    #    optimalbandwidth,
    #    vcovmatrix      
end 
