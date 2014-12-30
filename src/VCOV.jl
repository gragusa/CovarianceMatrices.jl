module VCOV

import StatsBase: confint, stderr, vcov, nobs
import GLM: LinearModel, GeneralizedLinearModel, ModelMatrix, df_residual
import DataFrames: DataFrameRegressionModel

using PDMats

const π²=π^2
const sixπ = 6*π

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       HC0, HC1, HC2, HC3, HC4,
       vcov, kernel, bread, meat

abstract RobustVariance
abstract HAC <: RobustVariance
abstract HC  <: RobustVariance

type HC0  <: HC end
type HC1  <: HC end
type HC2  <: HC end
type HC3  <: HC end
type HC4  <: HC end
type HC4m <: HC end
type HC5  <: HC end

include("HAC.jl")
include("optimalbw.jl")
include("HC.jl")

end # module
