module CovarianceMatrices

# using Requires
#
# function __init__()
#     @info("INIT CALLED")
#     @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("glm.jl")
# end

using LinearAlgebra
using Statistics

#include("varhac.jl")
include("types.jl")
include("HAC.jl")
include("HC.jl")
include("CRHC.jl")
include("covariance.jl")
include("methods.jl")
include("glm.jl")

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       TukeyHanningKernel, VARHAC, HC0, HC1, HC2, HC3, HC4, HC4m, HC5, CRHC0, CRHC1,
       CRHC2, CRHC3, NeweyWest, Andrews, 
       HACCache, HCCache, CRHCCache, CovarianceMatrix, vcov, stderr

end # module
