module CovarianceMatrices

# using Requires
#
# function __init__()
#     @info("INIT CALLED")
#     @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("glm.jl")
# end

<<<<<<< HEAD
=======
using PositiveFactorizations
>>>>>>> newcache
using LinearAlgebra
using Statistics

#include("varhac.jl")
include("types.jl")
include("HAC.jl")
include("HC.jl")
include("CRHC.jl")
<<<<<<< HEAD
include("covariance.jl")
include("methods.jl")
=======
>>>>>>> newcache
include("glm.jl")

export QuadraticSpectralKernel, TruncatedKernel, ParzenKernel, BartlettKernel,
       TukeyHanningKernel, VARHAC, HC0, HC1, HC2, HC3, HC4, HC4m, HC5, CRHC0, CRHC1,
<<<<<<< HEAD
       CRHC2, CRHC3, NeweyWest, Andrews, 
       HACCache, HCCache, CRHCCache, CovarianceMatrix, vcov, stderr
=======
       CRHC2, CRHC3, NeweyWest, Andrews, optimalbw, Variance,
       HACCache, HCCache, CRHCCache
>>>>>>> newcache

end # module
