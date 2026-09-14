"""
Tests for edge cases and error paths in CovarianceMatrices.jl.

This file tests error handling, numerical edge cases, and boundary conditions.
"""

using Test
using CovarianceMatrices
using LinearAlgebra
using StatsAPI
using Random
using DataFrames
using GLM

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

    @testset "Estimators are unchanged by use" begin
        k = Bartlett{Andrews}()
        X = randn(100, 2)

        V1 = aVar(k, X)
        @test CovarianceMatrices.bandwidth(V1) > 0

        # A second dataset selects its own bandwidth without disturbing the first
        # result or the estimator.
        bw1 = CovarianceMatrices.bandwidth(V1)
        V2 = aVar(k, X .* 2 .+ randn(100, 2))
        @test CovarianceMatrices.bandwidth(V1) == bw1
        @test k == Bartlett{Andrews}()

        # Reusing the estimator on the same data reproduces the same estimate.
        @test aVar(k, X) == V1
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
        struct EdgeBadModelNoCoef end
        @test_throws Exception CovarianceMatrices._check_coef(EdgeBadModelNoCoef())

        # Test _check_nobs error
        struct EdgeBadModelNoNobs end
        @test_throws Exception CovarianceMatrices._check_nobs(EdgeBadModelNoNobs())
    end

    @testset "Dimension checks for models" begin
        # Create a minimal MLikeModel for testing
        mutable struct EdgeTestMLike <: MLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            H::Matrix{Float64}
        end

        StatsAPI.coef(m::EdgeTestMLike) = m.theta
        StatsAPI.nobs(m::EdgeTestMLike) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::EdgeTestMLike) = m.Z
        CovarianceMatrices.hessian_objective(m::EdgeTestMLike) = m.H

        # Correctly identified model (m = k)
        m1 = EdgeTestMLike(randn(10, 3), randn(3), randn(3, 3))
        CovarianceMatrices._check_dimensions(Information(), m1)  # Should not throw

        # Misidentified model (m != k)
        m2 = EdgeTestMLike(randn(10, 4), randn(3), randn(3, 3))  # 4 moments, 3 params
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Information(), m2)
    end

    @testset "GMM dimension checks" begin
        # Create a minimal GMMLikeModel for testing
        mutable struct EdgeTestGMM <: GMMLikeModel
            Z::Matrix{Float64}
            theta::Vector{Float64}
            G::Matrix{Float64}
            H::Union{Nothing, Matrix{Float64}}
        end

        StatsAPI.coef(m::EdgeTestGMM) = m.theta
        StatsAPI.nobs(m::EdgeTestGMM) = size(m.Z, 1)
        CovarianceMatrices.momentmatrix(m::EdgeTestGMM) = m.Z
        CovarianceMatrices.jacobian_momentfunction(m::EdgeTestGMM) = m.G
        CovarianceMatrices.hessian_objective(m::EdgeTestGMM) = m.H

        # Overidentified GMM (m > k) - should work
        G = randn(4, 3)
        m1 = EdgeTestGMM(randn(10, 4), randn(3), G, randn(3, 3))
        CovarianceMatrices._check_dimensions(Information(), m1)

        # Underidentified GMM (m < k) - should throw
        G2 = randn(2, 3)
        m2 = EdgeTestGMM(randn(10, 2), randn(3), G2, randn(3, 3))
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Information(), m2)

        # GMM Misspecified without hessian - should throw
        m3 = EdgeTestGMM(randn(10, 4), randn(3), G, nothing)
        @test_throws ArgumentError CovarianceMatrices._check_dimensions(Misspecified(), m3)
    end

    @testset "aVar rejects non-real element types" begin
        # A complex moment matrix has no asymptotic-variance method and must fail
        # immediately rather than recursing.
        @test_throws MethodError aVar(HC0(), ComplexF64.(randn(20, 2)))
    end

    @testset "workingoptimalbw with a fixed bandwidth" begin
        X = randn(50, 2)
        Z, D, bw = CovarianceMatrices.workingoptimalbw(Bartlett(4), X)
        @test Z === X
        @test size(D) == (0, 0)
        @test eltype(D) == eltype(X)
        @test bw == 4.0
    end

    @testset "optimalbw with a fixed bandwidth" begin
        X = randn(50, 2)

        # A fixed bandwidth is part of the specification, so it is returned
        # unchanged and does not depend on the data or the keyword arguments.
        for 𝒦 in (Bartlett(4), Parzen(4), QuadraticSpectral(4),
            TukeyHanning(4), CovarianceMatrices.Truncated(4))
            @test optimalbw(𝒦, X) == 4.0
            @test optimalbw(𝒦, X; demean = true, prewhite = true) == 4.0
        end

        @test optimalbw(Bartlett(2.5), X) == 2.5
        @test optimalbw(Bartlett(4), X) == CovarianceMatrices.bandwidth(aVar(Bartlett(4), X))
    end

    @testset "demeaner" begin
        # `demeaner` operates on the moment matrix; a CR estimator is not a valid
        # first argument.
        @test_throws MethodError CovarianceMatrices.demeaner(CR0([1, 1, 2, 2]), randn(4, 2))
    end

    @testset "VARHAC symbol constructors" begin
        @test VARHAC(:aic) == VARHAC(AICSelector(), SameLags(8))
        @test VARHAC(:bic) == VARHAC(BICSelector(), SameLags(8))
        # Automatic lag selection is spelled `Val(:auto)`; a bare symbol names a
        # selector, so `:auto` is rejected with the list of valid selectors.
        @test VARHAC(Val(:auto)) == VARHAC(AICSelector(), AutoLags())
        @test_throws "Use :aic, :bic, or :fixed" VARHAC(:auto)
    end

    @testset "DriscollKraay identifier types" begin
        tis = repeat(1:5, inner = 4)
        iis = repeat(1:4, outer = 5)
        X = randn(20, 3)
        ref = aVar(DriscollKraay(Bartlett(2), tis = tis, iis = iis), X)

        # Identifiers of any type, and of differing types between the two
        # dimensions, are mapped to groups identically by both call forms.
        @test aVar(DriscollKraay(Bartlett(2), tis, iis), X) ≈ ref
        @test aVar(DriscollKraay(Bartlett(2), float.(tis), float.(iis)), X) ≈ ref
        @test aVar(DriscollKraay(Bartlett(2), float.(tis), iis), X) ≈ ref
        @test aVar(DriscollKraay(Bartlett(2), string.(tis), iis), X) ≈ ref
        @test aVar(
            DriscollKraay(
                Bartlett(2),
                CovarianceMatrices.Clustering(tis),
                CovarianceMatrices.Clustering(iis)
            ),
            X
        ) ≈ ref

        # Both index arrays are required; the estimator has no meaning without them.
        @test_throws "requires time indices" DriscollKraay(Bartlett(2))
        @test_throws "requires entity indices" DriscollKraay(Bartlett(2), tis = tis)
    end
end

@testset "scale switch and scaleby divisor" begin
    Random.seed!(20260915)
    X = randn(50, 3)
    n = size(X, 1)

    unscaled = parent(aVar(HC0(), X; scale = false))

    @testset "matrix entry point" begin
        # `scale=true` divides by the number of observations.
        @test parent(aVar(HC0(), X)) * n ≈ unscaled
        # `scaleby` divides by the value given, integer or float.
        @test parent(aVar(HC0(), X; scaleby = 97)) * 97 ≈ unscaled
        @test parent(aVar(HC0(), X; scaleby = 47.5)) * 47.5 ≈ unscaled
        # A divisor supersedes the switch rather than compounding with it.
        @test parent(aVar(HC0(), X; scale = true, scaleby = 97)) ≈
              parent(aVar(HC0(), X; scaleby = 97))
        @test parent(aVar(HC0(), X; scale = false, scaleby = 97)) ≈
              parent(aVar(HC0(), X; scaleby = 97))
    end

    @testset "invalid arguments" begin
        @test_throws "must be a positive finite number" aVar(HC0(), X; scaleby = 0)
        @test_throws "must be a positive finite number" aVar(HC0(), X; scaleby = -2.0)
        @test_throws "must be a positive finite number" aVar(HC0(), X; scaleby = Inf)
        @test_throws ArgumentError aVar(HC0(), X; scaleby = NaN)
        @test_throws "must be a `Bool`" aVar(HC0(), X; scale = :yes)
        @test_throws "pass the divisor as `scaleby` alone" aVar(
            HC0(), X; scale = 97, scaleby = 5)
    end

    @testset "deprecated numeric scale" begin
        # The old overloaded form still divides by the value it is given.
        deprecated = @test_deprecated aVar(HC0(), X; scale = 97)
        @test parent(deprecated) * 97 ≈ unscaled
    end

    @testset "regression entry points" begin
        y = X[:, 1]
        df = DataFrame(y = y, x = X[:, 2], z = X[:, 3], g = repeat(1:10, inner = 5))
        model = lm(@formula(y ~ x + z), df)

        for k in (HC1(), Bartlett(2), CR0(df.g))
            ref = parent(aVar(k, model; scale = false))
            @test parent(aVar(k, model)) * nobs(model) ≈ ref
            @test parent(aVar(k, model; scaleby = 43)) * 43 ≈ ref
        end
    end
end
