__precompile__(true)
module CovarianceMatrices

using Reexport
using PDMats

abstract type RobustVariance end
abstract type HAC{G} <: RobustVariance end
abstract type HC <: RobustVariance end
abstract type CRHC <: RobustVariance end

@reexport using GLM
@reexport using DataFrames

import StatsBase: confint, stderr, vcov, nobs, residuals
import GLM: LinPredModel, LinearModel, GeneralizedLinearModel, ModelMatrix, df_residual
import DataFrames: DataFrameRegressionModel

const fivetoπ² = 5.0 / π^2

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       TukeyHanningKernel, VARHAC, HC0, HC1, HC2, HC3, HC4, HC4m, HC5, CRHC0, CRHC1,
       CRHC2, CRHC3, vcov, NeweyWest, Andrews, optimalbw

mutable struct HC0  <: HC end
mutable struct HC1  <: HC end
mutable struct HC2  <: HC end
mutable struct HC3  <: HC end
mutable struct HC4  <: HC end
mutable struct HC4m <: HC end
mutable struct HC5  <: HC end

#const CLVector{T<:Integer} = DenseArray{T,1}

mutable struct CRHC0{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC1{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC2{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC3{V<:AbstractVector}  <: CRHC
    cl::V
end

include("varhac.jl")
include("HAC.jl")
include("optimalbw.jl")
include("HC.jl")

end # module
