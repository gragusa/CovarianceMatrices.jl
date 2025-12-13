"""
Tests for edge cases and error paths in CovarianceMatrices.jl.

This file tests error handling, numerical edge cases, and boundary conditions.
"""

using Test
using CovarianceMatrices
using LinearAlgebra
using StatsAPI

@testset "Edge Cases and Error Paths" begin
    @testset "ipinv edge cases" begin
        # Diagonal matrix path
        D = Diagonal([1.0, 2.0, 3.0])
        Dinv, flag, svals = CovarianceMatrices.ipinv(D)
        @test Dinv ≈ Diagonal([1.0, 0.5, 1/3])
        @test length(flag) == 3

        # Scalar
        @test CovarianceMatrices.ipinv(2.0) == 0.5
        @test CovarianceMatrices.ipinv(0.0) == 0.0
        @test CovarianceMatrices.ipinv(Inf) == 0.0

        # Empty matrix
        E = zeros(0, 0)
        Einv, eflag, esvals = CovarianceMatrices.ipinv(E)
        @test size(Einv) == (0, 0)

        # Near-singular matrix
        A = [1.0 0; 0 1e-16]
        Ainv, aflag, asvals = CovarianceMatrices.ipinv(A)
        @test size(Ainv) == (2, 2)

        # Regular matrix
        B = [1.0 0.5; 0.5 1.0]
        Binv, bflag, bsvals = CovarianceMatrices.ipinv(B)
        @test Binv ≈ inv(B) atol=1e-10
    end

    @testset "Debug output paths" begin
        # Test with debug=true to hit debug print paths
        A = [1.0 0; 0 1e-16]  # Near-singular
        result = CovarianceMatrices._compute_mle_information(A; debug = true, warn = false)
        @test size(result) == (2, 2)

        # Test MLE misspecified with debug
        H = [1.0 0.1; 0.1 1.0]
        Omega = [0.5 0.1; 0.1 0.5]
        V = CovarianceMatrices._compute_mle_misspecified(H, Omega; debug = true, warn = false)
        @test size(V) == (2, 2)
    end

    @testset "Kernel locking" begin
        # Test unlock_kernel! and lock_kernel!
        k = Bartlett{Andrews}()

        X = randn(100, 2)
        aVar(k, X)  # Sets bandwidth
        bw1 = k.bw[1]
        @test bw1 > 0

        # Lock and verify bandwidth doesn't change on repeated aVar call
        k.wlock .= true
        aVar(k, X .* 2)  # Different data
        @test k.bw[1] == bw1  # Bandwidth should not change when locked
    end

    @testset "aVar with various inputs" begin
        X = randn(50, 3)

        # Test with demeaning
        Σ1 = aVar(Bartlett(3), X; demean = true)
        Σ2 = aVar(Bartlett(3), X; demean = false)
        @test size(Σ1) == (3, 3)
        @test size(Σ2) == (3, 3)

        # Test with scaling
        Σ3 = aVar(Bartlett(3), X; scale = true)
        Σ4 = aVar(Bartlett(3), X; scale = false)
        @test size(Σ3) == (3, 3)
        @test size(Σ4) == (3, 3)

        # Test with prewhitening (for HAC)
        Σ5 = aVar(Bartlett(3), X; prewhite = true)
        @test size(Σ5) == (3, 3)
    end

    @testset "EWC with various bandwidths" begin
        X = randn(100, 2)

        for B in [1, 3, 5, 10]
            Σ = aVar(EWC(B), X)
            @test size(Σ) == (2, 2)
            @test issymmetric(Σ) || isapprox(Σ, Σ', atol = 1e-10)
        end
    end

    @testset "VARHAC with different strategies" begin
        X = randn(100, 2)

        # AIC with SameLags
        v1 = VARHAC(AICSelector(), SameLags(5))
        Σ1 = aVar(v1, X)
        @test size(Σ1) == (2, 2)

        # BIC with SameLags
        v2 = VARHAC(BICSelector(), SameLags(5))
        Σ2 = aVar(v2, X)
        @test size(Σ2) == (2, 2)

        # FixedLags
        v3 = VARHAC(FixedLags(3))
        Σ3 = aVar(v3, X)
        @test size(Σ3) == (2, 2)

        # AutoLags
        v4 = VARHAC(AICSelector(), AutoLags())
        Σ4 = aVar(v4, X)
        @test size(Σ4) == (2, 2)
    end

    @testset "Smoothed moments estimators" begin
        X = randn(100, 2)

        # UniformSmoother
        us = UniformSmoother(5)
        Σ1 = aVar(us, X)
        @test size(Σ1) == (2, 2)

        # TriangularSmoother
        ts = TriangularSmoother(5)
        Σ2 = aVar(ts, X)
        @test size(Σ2) == (2, 2)
    end

    @testset "Model interface validation" begin
        # Test _check_coef error
        struct BadModelNoCoef end
        @test_throws Exception CovarianceMatrices._check_coef(BadModelNoCoef())

        # Test _check_nobs error
        struct BadModelNoNobs end
        @test_throws Exception CovarianceMatrices._check_nobs(BadModelNoNobs())
    end

    @testset "Dimension checks for models" begin
        # Create a minimal MLikeModel for testing
        mutable struct TestMLike <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::TestMLike) = m.theta
        StatsAPI.nobs(m::TestMLike) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::TestMLike) = m.Z
        CovarianceMatrices.hessian_objective(m::TestMLike) = m.H

        # Correctly identified model (m = k)
        m1 = TestMLike(randn(10, 3), randn(3), randn(3, 3))
        CovarianceMatrices._check_dimensions(Information(), m1)  # Should not throw

        # Misidentified model (m != k)
        m2 = TestMLike(randn(10, 4), randn(3), randn(3, 3))  # 4 moments, 3 params
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Information(), m2)
    end

    @testset "GMM dimension checks" begin
        # Create a minimal GMMLikeModel for testing
        mutable struct TestGMM <: GMMLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            G::Matrix{Float64}
            H::Union{Nothing, Matrix{Float64}}
        end

        StatsAPI.coef(m::TestGMM) = m.theta
        StatsAPI.nobs(m::TestGMM) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::TestGMM) = m.Z
        CovarianceMatrices.jacobian_momentfunction(m::TestGMM) = m.G
        CovarianceMatrices.hessian_objective(m::TestGMM) = m.H

        # Overidentified GMM (m > k) - should work
        G = randn(4, 3)
        m1 = TestGMM(randn(10, 4), randn(3), G, randn(3, 3))
        CovarianceMatrices._check_dimensions(Information(), m1)

        # Underidentified GMM (m < k) - should throw
        G2 = randn(2, 3)
        m2 = TestGMM(randn(10, 2), randn(3), G2, randn(3, 3))
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Information(), m2)

        # GMM Misspecified without hessian - should throw
        m3 = TestGMM(randn(10, 4), randn(3), G, nothing)
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Misspecified(), m3)
    end
end
