"""
Tests for API coverage in CovarianceMatrices.jl.

This file tests the unified API (api.jl) and model interface (model_interface.jl).
"""

using Test
using CovarianceMatrices
using LinearAlgebra
using StatsAPI

@testset "API Coverage" begin

    @testset "stderror wrapper" begin
        # Create a minimal MLikeModel for testing
        mutable struct StderrTestModel <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::StderrTestModel) = m.theta
        StatsAPI.nobs(m::StderrTestModel) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::StderrTestModel) = m.Z
        CovarianceMatrices.hessian_objective(m::StderrTestModel) = m.H

        # Create a valid model
        n, k = 50, 3
        Z = randn(n, k)
        H = Z' * Z  # Positive definite Hessian
        theta = randn(k)
        model = StderrTestModel(Z, theta, H)

        # Test stderror returns sqrt of diagonal of vcov
        # Use HC0 which has a working avar method
        V = vcov(HC0(), Information(), model)
        se = stderror(HC0(), Information(), model)

        @test length(se) == k
        @test se ≈ sqrt.(diag(V))
        @test all(se .>= 0)

        # Test with Misspecified form (needs an estimator with avar method)
        se_mis = stderror(HC0(), Misspecified(), model)
        @test length(se_mis) == k
        @test all(se_mis .>= 0)
    end

    @testset "cross_score default implementation" begin
        # Verify default cross_score returns g'g
        mutable struct CrossScoreTestModel <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
        end

        StatsAPI.coef(m::CrossScoreTestModel) = m.theta
        StatsAPI.nobs(m::CrossScoreTestModel) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::CrossScoreTestModel) = m.Z
        CovarianceMatrices.hessian_objective(m::CrossScoreTestModel) = m.Z' * m.Z

        model = CrossScoreTestModel(randn(20, 3), randn(3))
        Z = CovarianceMatrices.momentmatrix(model)
        G = CovarianceMatrices.cross_score(model)

        @test G ≈ Z' * Z
        @test size(G) == (3, 3)
        @test issymmetric(G)
    end

    @testset "jacobian_momentfunction default" begin
        # Default returns nothing for generic models
        struct JacTestModel end
        @test CovarianceMatrices.jacobian_momentfunction(JacTestModel()) === nothing
    end

    @testset "weight_matrix interface" begin
        # weight_matrix is defined but returns nothing by default
        struct WeightTestModel2 end
        # The default hessian_objective just returns nothing for unknown types
        @test CovarianceMatrices.weight_matrix(WeightTestModel2()) === nothing
    end

    @testset "Matrix compatibility checks" begin
        # Test _check_matrix_compatibility for Information form
        Z = randn(10, 3)

        # Valid cross_score
        cross_score_valid = Z' * Z
        CovarianceMatrices._check_matrix_compatibility(
            Information(), Z, cross_score_valid, nothing, nothing
        )

        # Valid hessian_objective
        H_valid = randn(3, 3)
        CovarianceMatrices._check_matrix_compatibility(
            Information(), Z, nothing, H_valid, nothing
        )

        # Error: both nothing
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Information(), Z, nothing, nothing, nothing
        )

        # Error: non-square hessian
        H_bad = randn(3, 4)
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Information(), Z, nothing, H_bad, nothing
        )

        # Error: wrong size cross_score
        cross_score_bad = randn(2, 2)
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Information(), Z, cross_score_bad, nothing, nothing
        )
    end

    @testset "Matrix compatibility - Misspecified form" begin
        Z = randn(10, 3)
        cross_score_valid = Z' * Z

        # Valid case
        CovarianceMatrices._check_matrix_compatibility(
            Misspecified(), Z, cross_score_valid, nothing, nothing
        )

        # Error: cross_score required
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Misspecified(), Z, nothing, nothing, nothing
        )

        # Error: wrong size cross_score
        cross_score_bad = randn(2, 2)
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Misspecified(), Z, cross_score_bad, nothing, nothing
        )

        # Error: wrong size weight matrix
        W_bad = randn(2, 2)
        @test_throws ArgumentError CovarianceMatrices._check_matrix_compatibility(
            Misspecified(), Z, cross_score_valid, nothing, W_bad
        )

        # Valid with weight matrix
        W_valid = randn(3, 3)
        CovarianceMatrices._check_matrix_compatibility(
            Misspecified(), Z, cross_score_valid, nothing, W_valid
        )
    end

    @testset "GMM vcov with weight matrix W" begin
        # Create a GMMLikeModel for testing weighted GMM
        mutable struct WeightedGMMModel2 <: GMMLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            G::Matrix{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::WeightedGMMModel2) = m.theta
        StatsAPI.nobs(m::WeightedGMMModel2) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::WeightedGMMModel2) = m.Z
        CovarianceMatrices.jacobian_momentfunction(m::WeightedGMMModel2) = m.G
        CovarianceMatrices.hessian_objective(m::WeightedGMMModel2) = m.H

        # Create overidentified GMM (m > k)
        n, m_moments, k = 100, 4, 3
        Z = randn(n, m_moments)
        G = randn(m_moments, k)  # Jacobian: m x k
        H = randn(k, k)
        H = H' * H  # Make positive definite
        theta = randn(k)

        model = WeightedGMMModel2(Z, theta, G, H)

        # Test Information form with custom W (use HC0 which has avar)
        W = Matrix{Float64}(I, m_moments, m_moments)
        V_weighted = vcov(HC0(), Information(), model; W = W)
        @test size(V_weighted) == (k, k)

        # Test without W (optimal GMM)
        V_optimal = vcov(HC0(), Information(), model)
        @test size(V_optimal) == (k, k)

        # Test Misspecified form with W
        V_mis_weighted = vcov(HC0(), Misspecified(), model; W = W)
        @test size(V_mis_weighted) == (k, k)

        # Test Misspecified form without W
        V_mis_optimal = vcov(HC0(), Misspecified(), model)
        @test size(V_mis_optimal) == (k, k)
    end

    @testset "MLikeModel without hessian for Misspecified" begin
        # Test that MLikeModel without hessian_objective throws for Misspecified
        mutable struct NoHessianMLE2 <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
        end

        StatsAPI.coef(m::NoHessianMLE2) = m.theta
        StatsAPI.nobs(m::NoHessianMLE2) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::NoHessianMLE2) = m.Z
        CovarianceMatrices.hessian_objective(m::NoHessianMLE2) = nothing

        model = NoHessianMLE2(randn(20, 3), randn(3))

        # Should throw for Misspecified form
        @test_throws ArgumentError vcov(HC0(), Misspecified(), model)

        # Information form should work (falls back to cross_score)
        V = vcov(HC0(), Information(), model)
        @test size(V) == (3, 3)
    end

    @testset "vcov with check=false" begin
        # Test that check=false skips validation
        mutable struct SkipCheckModel2 <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::SkipCheckModel2) = m.theta
        StatsAPI.nobs(m::SkipCheckModel2) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::SkipCheckModel2) = m.Z
        CovarianceMatrices.hessian_objective(m::SkipCheckModel2) = m.H

        n, k = 30, 3
        Z = randn(n, k)
        H = Z' * Z
        model = SkipCheckModel2(Z, randn(k), H)

        # Should work with check=false
        V = vcov(HC0(), Information(), model; check = false)
        @test size(V) == (k, k)

        # Also with debug and warn options
        V2 = vcov(HC0(), Information(), model; debug = false, warn = false)
        @test size(V2) == (k, k)
    end

    @testset "vcov tolerance parameters" begin
        # Test cond_atol and cond_rtol parameters
        mutable struct TolTestModel2 <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::TolTestModel2) = m.theta
        StatsAPI.nobs(m::TolTestModel2) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::TolTestModel2) = m.Z
        CovarianceMatrices.hessian_objective(m::TolTestModel2) = m.H

        n, k = 30, 3
        Z = randn(n, k)
        H = Z' * Z
        model = TolTestModel2(Z, randn(k), H)

        # Test with custom tolerances
        V1 = vcov(HC0(), Information(), model; cond_atol = 1e-10)
        V2 = vcov(HC0(), Information(), model; cond_rtol = 1e-8)
        V3 = vcov(HC0(), Information(), model; cond_atol = 1e-10, cond_rtol = 1e-8)

        @test size(V1) == (k, k)
        @test size(V2) == (k, k)
        @test size(V3) == (k, k)
    end

    @testset "HAC estimator with models" begin
        # Test HAC works with model interface
        mutable struct HACTestModel <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::HACTestModel) = m.theta
        StatsAPI.nobs(m::HACTestModel) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::HACTestModel) = m.Z
        CovarianceMatrices.hessian_objective(m::HACTestModel) = m.H

        n, k = 50, 2
        Z = randn(n, k)
        H = Z' * Z
        model = HACTestModel(Z, randn(k), H)

        # Test with Bartlett kernel
        V = vcov(Bartlett(3), Misspecified(), model)
        @test size(V) == (k, k)
        @test isapprox(V, V', atol=1e-10)

        # Test with Parzen kernel
        V2 = vcov(Parzen{Andrews}(), Misspecified(), model)
        @test size(V2) == (k, k)
    end

end
