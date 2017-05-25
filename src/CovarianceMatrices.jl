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


const π²=π^2
const sixπ = 6*π

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       TukeyHanningKernel, VARHAC, HC0, HC1, HC2, HC3, HC4, HC4m, HC5, CRHC0, CRHC1,
       CRHC2, CRHC3, vcov, NeweyWest, Andrews, optimalbw

type HC0  <: HC end
type HC1  <: HC end
type HC2  <: HC end
type HC3  <: HC end
type HC4  <: HC end
type HC4m <: HC end
type HC5  <: HC end

#const CLVector{T<:Integer} = DenseArray{T,1}

type CRHC0{V<:AbstractVector}  <: CRHC
    cl::V
end

type CRHC1{V<:AbstractVector}  <: CRHC
    cl::V
end

type CRHC2{V<:AbstractVector}  <: CRHC
    cl::V
end

type CRHC3{V<:AbstractVector}  <: CRHC
    cl::V
end

include("varhac.jl")
include("HAC.jl")
include("optimalbw.jl")
include("HC.jl")

end # module
