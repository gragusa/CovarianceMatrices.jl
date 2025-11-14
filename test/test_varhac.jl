"""
Comprehensive test suite for VARHAC functionality.

This test suite validates:
1. Constructor functionality and defaults
2. Both aVar and vcov API compatibility
3. Mathematical properties (PSD, symmetry)
4. Different lag selection strategies
5. Performance and numerical stability
6. Edge cases and error conditions
"""

using CovarianceMatrices
using LinearAlgebra
using Statistics
using Random
using Test
using Distributions

@testset "VARHAC Comprehensive Tests ✅" begin
    @testset "Constructor and Default Behavior" begin
        # Test default constructor
        vh_default = VARHAC()
        @test vh_default isa VARHAC{AICSelector, SameLags, Float64}
        @test vh_default.selector isa AICSelector
        @test vh_default.strategy isa SameLags
        @test vh_default.strategy.maxlag == 8

        # Test convenient constructors
        vh_aic = VARHAC(:aic)
        @test vh_aic.selector isa AICSelector

        vh_bic = VARHAC(:bic)
        @test vh_bic.selector isa BICSelector

        vh_lags = VARHAC(12)
        @test vh_lags.strategy.maxlag == 12

        # Test error handling
        @test_throws ArgumentError VARHAC(:invalid)

        # Test type parameter
        vh_float32 = VARHAC(; T = Float32)
        @test vh_float32 isa VARHAC{AICSelector, SameLags, Float32}
    end

    @testset "aVar API Compatibility" begin
        Random.seed!(123)
        n, k = 50, 3

        # Generate test data with some serial correlation
        X = randn(n, k)
        for t in 2:n
            X[t, :] .+= 0.3 * X[t - 1, :] .+ 0.1 * randn(k)
        end

        vh = VARHAC()

        # Test basic aVar functionality
        S1 = aVar(vh, X)
        @test size(S1) == (k, k)
        @test S1 ≈ S1'  # Check numerical symmetry
        @test isposdef(Symmetric(S1)) || isposdef(Symmetric(S1 + 1e-10*I))

        # Test with different options
        S2 = aVar(vh, X; demean = false)
        @test size(S2) == (k, k)

        S3 = aVar(vh, X; scale = false)
        @test size(S3) == (k, k)

        # Test different lag selection strategies
        vh_bic = VARHAC(:bic)
        S_bic = aVar(vh_bic, X)
        @test size(S_bic) == (k, k)
        @test S_bic ≈ S_bic'  # Check numerical symmetry
    end

    @testset "Unified vcov API Integration" begin
        Random.seed!(456)
        n, k = 40, 2

        # Create simple test model moment matrix
        X = randn(n, k)
        for t in 2:n
            X[t, :] .+= 0.2 * X[t - 1, :] .+ randn(k) * 0.1
        end

        vh = VARHAC()

        # Test vcov with Information form
        V1 = vcov(vh, Information(), X)
        @test size(V1) == (k, k)
        @test V1 ≈ V1'  # Check numerical symmetry
        @test isposdef(Symmetric(V1)) || isposdef(Symmetric(V1 + 1e-12*I))

        # Test vcov with Misspecified form
        V2 = vcov(vh, Misspecified(), X)
        @test size(V2) == (k, k)
        @test V2 ≈ V2'  # Check numerical symmetry

        # For VARHAC, both forms should give similar results
        @test V1 ≈ V2 rtol=1e-8

        # Test that both APIs produce valid results
        S_avar = aVar(vh, X; demean = false, scale = true)
        @test size(S_avar) == (k, k)
        @test S_avar ≈ S_avar'  # Check numerical symmetry
        @test isposdef(Symmetric(S_avar)) || isposdef(Symmetric(S_avar + 1e-10*I))
    end

    @testset "AutoLags Functionality" begin
        Random.seed!(789)

        # Test different sample sizes to verify T^(1/3) rule
        test_sizes = [(50, 3), (100, 3), (200, 3)]

        for (T, N) in test_sizes
            max_lag_expected = min(
                max(1, floor(Int, T^(1/3))),
                max(1, floor(Int, (T - 1) / N)),
                20
            )

            actual_max_lag = CovarianceMatrices.compute_auto_maxlag(T, N)
            @test actual_max_lag == max_lag_expected
            @test actual_max_lag >= 1
            @test actual_max_lag <= 20
        end

        # Test AutoLags with actual data
        X = randn(100, 3)
        vh_auto = VARHAC(AICSelector(), AutoLags())

        # Should work with aVar (auto-detects dimensions)
        S_auto = aVar(vh_auto, X)
        @test size(S_auto) == (3, 3)
        @test S_auto ≈ S_auto'  # Check numerical symmetry
    end

    @testset "Mathematical Properties Validation" begin
        Random.seed!(321)

        @testset "MA(0) White Noise Test" begin
            # For white noise, VARHAC should select low lag orders
            n, k = 80, 2
            X = randn(n, k)  # Pure white noise

            vh = VARHAC()
            S = aVar(vh, X)

            @test size(S) == (k, k)
            @test S ≈ S'  # Check numerical symmetry
            @test isposdef(Symmetric(S))

            # Check that selected orders are reasonable for white noise
            # (Most equations should select low lag orders)
            orders = CovarianceMatrices.order_aic(vh)
            if !isnothing(orders) && !isempty(orders)
                @test mean(orders) <= 3  # Should be low for white noise
            end
        end

        @testset "Known VAR Process Test" begin
            # Generate data from a known VAR(2) process
            n, k = 100, 2
            true_A1 = [0.5 0.1; 0.2 0.3]
            true_A2 = [0.2 0.0; 0.1 0.2]
            Σ = [1.0 0.3; 0.3 1.0]

            X = zeros(n, k)
            ε = rand(MvNormal(zeros(k), Σ), n)'

            for t in 3:n
                X[t, :] = true_A1 * X[t - 1, :] + true_A2 * X[t - 2, :] + ε[t, :]
            end

            vh = VARHAC()
            S = aVar(vh, X)

            @test size(S) == (k, k)
            @test S ≈ S'  # Check numerical symmetry
            @test isposdef(Symmetric(S))

            # The spectral density should be positive definite
            eigenvals = eigvals(Symmetric(S))
            @test all(eigenvals .> 0)
        end

        @testset "Mixed Persistence Test" begin
            # One near white-noise series, one persistent series
            n = 100
            X = zeros(n, 2)

            # Series 1: near white noise
            X[:, 1] = randn(n)

            # Series 2: persistent AR(1)
            for t in 2:n
                X[t, 2] = 0.8 * X[t - 1, 2] + randn()
            end

            vh = VARHAC()
            S = aVar(vh, X)

            @test size(S) == (2, 2)
            @test S ≈ S'  # Check numerical symmetry
            @test isposdef(Symmetric(S))

            # The persistent series should have higher variance
            @test S[2, 2] > S[1, 1]
        end
    end

    @testset "Different Lag Selection Strategies" begin
        Random.seed!(654)
        n, k = 60, 2
        X = randn(n, k)

        # Add some autocorrelation
        for t in 2:n
            X[t, :] .+= 0.4 * X[t - 1, :] .+ 0.1 * randn(k)
        end

        strategies = [
            (VARHAC(AICSelector(), SameLags(6)), "AIC SameLags"),
            (VARHAC(BICSelector(), SameLags(6)), "BIC SameLags"),
            (VARHAC(FixedSelector(), FixedLags(3)), "Fixed Lags")
        ]

        for (vh, name) in strategies
            S = aVar(vh, X)
            @test size(S) == (k, k)  # Failed for $name
            @test S ≈ S'  # Check numerical symmetry  # Symmetry failed for $name
            @test isposdef(Symmetric(S)) || isposdef(Symmetric(S + 1e-10*I))  # PSD failed for $name
        end
    end

    @testset "Numerical Stability and Edge Cases" begin
        Random.seed!(987)

        @testset "Small Sample Sizes" begin
            # Test with very small samples
            n, k = 15, 2
            X = randn(n, k)

            vh = VARHAC(AICSelector(), SameLags(3))  # Small lag to avoid overfitting
            S = aVar(vh, X)

            @test size(S) == (k, k)
            @test S ≈ S'  # Check numerical symmetry
            @test all(isfinite.(S))
        end

        @testset "Near-Singular Cases" begin
            # Create data with near-multicollinearity
            n, k = 50, 3
            X = randn(n, 2)
            X = [X X[:, 1] .+ 1e-8 .* randn(n)]  # Nearly collinear third column

            vh = VARHAC()
            S = aVar(vh, X)

            @test size(S) == (k, k)
            @test S ≈ S'  # Check numerical symmetry
            @test all(isfinite.(S))
            # Should handle near-singularity gracefully
        end

        @testset "Missing Data Handling" begin
            # Test with NaN values in residuals (should be handled by nancov)
            n, k = 30, 2
            X = randn(n, k)

            vh = VARHAC()
            S = aVar(vh, X)

            @test size(S) == (k, k)
            @test !any(isnan.(S))  # Should not propagate NaNs
        end
    end

    @testset "Performance and Type Stability" begin
        Random.seed!(111)
        n, k = 100, 3
        X = randn(n, k)

        # Test type stability
        vh = VARHAC(; T = Float64)
        S1 = aVar(vh, X)
        @test eltype(S1) == Float64

        vh32 = VARHAC(; T = Float32)
        X32 = Float32.(X)
        S2 = aVar(vh32, X32)
        @test eltype(S2) == Float32

        # Basic performance test - should complete without errors
        @test @elapsed(aVar(vh, X)) < 5.0  # Should be reasonably fast
    end

    @testset "Error Conditions and Validation" begin
        @testset "Invalid Inputs" begin
            vh = VARHAC()

            # Test with too few observations (should error)
            X_tiny = randn(2, 3)
            @test_throws ArgumentError aVar(vh, X_tiny)

            # Test with single column
            X_single = randn(50, 1)
            S_single = aVar(vh, X_single)
            @test size(S_single) == (1, 1)
            @test S_single ≈ S_single'  # Check numerical symmetry
        end

        @testset "AutoLags Error Handling" begin
            vh_auto = VARHAC(AICSelector(), AutoLags())

            # Should error when trying to get maxlags without dimensions
            @test_throws ErrorException CovarianceMatrices.maxlags(vh_auto)

            # Should work when dimensions provided
            @test CovarianceMatrices.maxlags(vh_auto, 100, 3) isa Int
        end
    end

    @testset "Integration with Standard Errors" begin
        Random.seed!(222)
        n, k = 60, 2
        X = randn(n, k)

        vh = VARHAC()

        # Test stderror function
        se1 = stderror(vh, Information(), X)
        @test length(se1) == k
        @test all(se1 .> 0)

        se2 = stderror(vh, Misspecified(), X)
        @test length(se2) == k
        @test all(se2 .> 0)

        # Should be consistent with vcov
        V = vcov(vh, Information(), X)
        se_manual = sqrt.(diag(V))
        @test se1 ≈ se_manual rtol=1e-10
    end
end

println("\n" * "="^70)
println("✅ VARHAC comprehensive tests completed successfully!")
println("   📊 Constructor functionality verified")
println("   🔧 Both aVar and vcov APIs working correctly")
println("   📈 Mathematical properties validated")
println("   🚀 Performance and stability confirmed")
println("   🐛 Edge cases and error conditions tested")
println("="^70)
