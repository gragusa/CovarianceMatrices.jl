__precompile__(true)
module CovarianceMatrices

using Reexport
using PDMats

abstract RobustVariance
abstract HAC{G}  <: RobustVariance
abstract HC   <: RobustVariance
abstract CRHC <: RobustVariance

@reexport using GLM
@reexport using DataFrames
#@reexport using InstrumentalVariables

import StatsBase: confint, stderr, vcov, nobs, residuals
import GLM: LinPredModel, LinearModel, GeneralizedLinearModel, ModelMatrix, df_residual
import DataFrames: DataFrameRegressionModel
#import InstrumentalVariables: IVResp, LinearIVModel, residuals

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

typealias CLVector{T<:Integer} DenseArray{T,1}

type CRHC0{V<:CLVector}  <: CRHC
    cl::V
end

type CRHC1{V<:CLVector}  <: CRHC
    cl::V
end

type CRHC2{V<:CLVector}  <: CRHC
    cl::V
end

type CRHC3{V<:CLVector}  <: CRHC
    cl::V
end

include("varhac.jl")
include("HAC.jl")
include("optimalbw.jl")
include("HC.jl")

end # module
