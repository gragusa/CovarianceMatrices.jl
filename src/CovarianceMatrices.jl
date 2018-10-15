module CovarianceMatrices

using LinearAlgebra
using Statistics
using Requires
using PositiveFactorizations

function __init___()
    @info("INIT CALLED")
    @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("glm.jl")
end




#include("varhac.jl")
include("types.jl")
include("HAC.jl")
include("HC.jl")
include("CRHC.jl")

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       TukeyHanningKernel, VARHAC, HC0, HC1, HC2, HC3, HC4, HC4m, HC5, CRHC0, CRHC1,
       CRHC2, CRHC3, NeweyWest, Andrews, optimalbw, Variance,
       HACCache, HCCache, CRHCCache

end # module
