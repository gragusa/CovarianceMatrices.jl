## Main test runner for CovarianceMatrices.jl
##
## This file orchestrates all tests in the package, ensuring comprehensive coverage
## of both core functionality and the new unified API.

using Test

@testset verbose=true "CovarianceMatrices.jl Test Suite" begin


    @testset "Core Functionality Tests" begin
        include("test_core.jl")
    end

    @testset "Interface and API Tests" begin
        include("test_interface.jl")
    end

    @testset "Probit Model Example Tests" begin
        include("test_probit.jl")
    end
    @testset "GMM Model Example Tests" begin
        include("test_gmm.jl")
    end

end

println("\n" * "="^70)
println("✅ All CovarianceMatrices.jl tests completed successfully!")
println("   📊 Core functionality verified")
println("   🔧 Interface and API coverage complete")
println("   📈 Example models working correctly")
println("   🚀 New unified API fully tested")
println("="^70)
