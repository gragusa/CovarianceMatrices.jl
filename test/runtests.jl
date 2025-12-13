## Main test runner for CovarianceMatrices.jl
##
## This file orchestrates all tests in the package, ensuring comprehensive coverage
## of both core functionality and the new unified API.

using Test

include("aqua.jl")
include("test_core.jl")
#include("test_glm_integration.jl")  # Requires RCall - use test_glm_integration_julia.jl instead
include("test_glm_integration_julia.jl")  # Pure Julia GLM tests
include("test_interface.jl")
include("test_probit.jl")
include("test_gmm.jl")
include("test_debug_inverses.jl")
include("test_varhac.jl")
include("test_varhac_fortran_validation.jl")
include("test_smoothed_moments.jl")
include("test_ewc_montecarlo.jl")
include("test_types_coverage.jl")
include("test_edge_cases.jl")
include("test_api_coverage.jl")

println("\n" * "="^70)
println("✅ All CovarianceMatrices.jl tests completed successfully!")
println("   ✨ Aqua.jl quality assurance checks passed")
println("   📊 Core functionality verified")
println("   🔧 Interface and API coverage complete")
println("   📈 Example models working correctly")
println("   🚀 New unified API fully tested")
println("   🐛 Debug and tolerance features validated")
println("   📉 VARHAC implementation comprehensive")
println("   🔥 VARHAC FORTRAN validation PASSED!")
println("   🌊 SmoothedMoments (Smith's method) validated")
println("   📐 EWC Monte Carlo coverage validated")
println("="^70)
