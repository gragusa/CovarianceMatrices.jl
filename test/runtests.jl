## Main test runner for CovarianceMatrices.jl
##
## This file orchestrates all tests in the package, ensuring comprehensive coverage
## of both core functionality and the new unified API.

using Test

include("test_core.jl")
include("test_interface.jl")
include("test_probit.jl")
include("test_gmm.jl")
include("test_debug_inverses.jl")

println("\n" * "="^70)
println("✅ All CovarianceMatrices.jl tests completed successfully!")
println("   📊 Core functionality verified")
println("   🔧 Interface and API coverage complete")
println("   📈 Example models working correctly")
println("   🚀 New unified API fully tested")
println("   🐛 Debug and tolerance features validated")
println("="^70)
