"""
Pure Julia GLM integration tests for CovarianceMatrices.jl.

These tests verify GLM integration without requiring RCall.
Expected values are pre-computed from R's sandwich package.
"""

using Test
using CovarianceMatrices
using CategoricalArrays
using DataFrames
using GLM
using StableRNGs
using LinearAlgebra

# Helper function for approximate symmetry check
is_approx_symmetric(V; atol = 1e-10) = isapprox(V, V', atol = atol)

@testset "GLM Integration (Pure Julia)" begin
    @testset "Linear Model - momentmatrix" begin
        rng = StableRNG(1234)
        df = DataFrame(X1 = randn(rng, 10), X2 = randn(rng, 10), Y = randn(rng, 10))
        m = lm(@formula(Y~X1 + X2), df)
        M = momentmatrix(m)

        # Expected values computed from R's sandwich::estfun()
        expected = [-0.44514751 -0.23125171 0.391807752;
                    -0.04562477 -0.04129679 -0.076279862;
                    -0.71538836 1.19874849 0.104035222;
                    0.08256957 -0.10690558 -0.033658338;
                    0.59596497 -0.40785784 0.318478340;
                    0.49708819 -0.34107733 -0.401033205;
                    1.05956405 0.09773035 0.446067578;
                    -1.14182761 0.10235007 -0.769137251;
                    -0.12131377 0.17616015 0.007348568;
                    0.23411525 -0.44659979 0.012371195]
        @test M ≈ expected atol=1e-6
    end

    @testset "Linear Model - HAC variance" begin
        rng = StableRNG(1234)
        df = DataFrame(X1 = randn(rng, 10), X2 = randn(rng, 10), Y = randn(rng, 10))
        m = lm(@formula(Y~X1 + X2), df)

        K = Bartlett{NeweyWest}()
        Σ = aVar(K, m)
        @test K.bw[1] ≈ 1.737626 atol=1e-06

        # Test that bandwidth is set after aVar call
        @test K.bw[1] > 0
    end

    @testset "Linear Model - HC estimators" begin
        rng = StableRNG(1234)
        df = DataFrame(
            X1 = randn(rng, 20),
            X2 = randn(rng, 20),
            Y = randn(rng, 20),
            w = rand(rng, 20)
        )
        m = lm(@formula(Y~X1 + X2), df)

        # Test all HC estimators produce valid results
        for hc in [HC0(), HC1(), HC2(), HC3(), HC4(), HC5()]
            V = vcov(hc, m)
            @test size(V) == (3, 3)
            @test is_approx_symmetric(V)
            @test all(diag(V) .>= 0)
        end
    end

    @testset "Weighted Linear Model" begin
        rng = StableRNG(1234)
        df = DataFrame(
            X1 = randn(rng, 20),
            X2 = randn(rng, 20),
            Y = randn(rng, 20),
            w = rand(rng, 20)
        )
        m = lm(@formula(Y~X1 + X2), df, wts = df.w)

        # Test residuals for weighted model
        r = CovarianceMatrices._residuals(m.model)
        @test length(r) == 20

        # Test momentmatrix for weighted model
        M = momentmatrix(m)
        @test size(M) == (20, 3)

        # Test HC estimators with weights
        for hc in [HC0(), HC1(), HC2(), HC3()]
            V = vcov(hc, m)
            @test size(V) == (3, 3)
            @test all(diag(V) .>= 0)
        end
    end

    @testset "Poisson GLM" begin
        # Standard Poisson example
        counts = [18, 17, 15, 20, 10, 20, 25, 13, 12]
        outcome = repeat(1:3, 3)
        treatment = repeat(1:3, inner = 3)
        df = DataFrame(
            treatment = categorical(treatment),
            outcome = categorical(outcome),
            counts = counts
        )

        glm_model = glm(@formula(counts ~ outcome + treatment), df, Poisson())

        # Test momentmatrix
        M = momentmatrix(glm_model)
        @test size(M, 1) == 9
        @test size(M, 2) == length(coef(glm_model))

        # Test HC estimators
        for hc in [HC0(), HC1(), HC2(), HC3()]
            V = vcov(hc, glm_model)
            @test size(V, 1) == length(coef(glm_model))
            @test is_approx_symmetric(V)
            @test all(diag(V) .>= 0)
        end
    end

    @testset "Weighted Poisson GLM" begin
        counts = [18, 17, 15, 20, 10, 20, 25, 13, 12]
        outcome = repeat(1:3, 3)
        treatment = repeat(1:3, inner = 3)
        weights = [1.0, 0.5, 1.5, 1.0, 0.8, 1.2, 1.0, 0.9, 1.1]
        df = DataFrame(
            treatment = categorical(treatment),
            outcome = categorical(outcome),
            counts = counts,
            weights = weights
        )

        glm_model = glm(@formula(counts ~ outcome + treatment), df, Poisson(), wts = df.weights)

        # Test momentmatrix for weighted GLM
        M = momentmatrix(glm_model)
        @test size(M, 1) == 9

        # Test HC estimators with weighted GLM
        for hc in [HC0(), HC1()]
            V = vcov(hc, glm_model)
            @test size(V, 1) == length(coef(glm_model))
            @test all(diag(V) .>= 0)
        end
    end

    @testset "Gamma GLM" begin
        clotting = DataFrame(
            u = log.([5, 10, 15, 20, 30, 40, 60, 80, 100]),
            lot1 = [118, 58, 42, 35, 27, 25, 21, 19, 18],
            lot2 = [69, 35, 26, 21, 18, 16, 13, 12, 12],
            w = 9.0 * [1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
        )

        GAMMA = glm(
            @formula(lot1 ~ u),
            clotting,
            Gamma(),
            InverseLink(),
            wts = convert(Array, clotting[!, :w])
        )

        # Test HAC variance
        k = Parzen{Andrews}()
        V = vcov(k, GAMMA)
        bw = k.bw[1]

        @test size(V) == (2, 2)
        @test bw > 0

        # Pre-computed expected variance from R
        Vp = [5.48898e-7 -2.60409e-7; -2.60409e-7 1.4226e-7]
        @test V ≈ Vp atol=1e-08
    end

    @testset "HAC with GLM" begin
        rng = StableRNG(1234)
        df = DataFrame(count = abs.(round.(Int, 10.0 * randn(rng, 100))), X2 = randn(rng, 100))
        dum = glm(@formula(count ~ X2), df, Poisson())

        # Test various HAC kernels
        for kernel in [Bartlett{Andrews}(), Parzen{Andrews}(), QuadraticSpectral{Andrews}()]
            V = vcov(kernel, dum, dofadjust = false)
            @test size(V) == (2, 2)
            @test is_approx_symmetric(V)
            @test all(diag(V) .>= 0)
        end

        # Test with DOF adjustment
        V_adj = vcov(Bartlett{Andrews}(), dum, dofadjust = true)
        V_noadj = vcov(Bartlett{Andrews}(), dum, dofadjust = false)
        # DOF adjustment should increase variance
        @test all(diag(V_adj) .>= diag(V_noadj))
    end

    @testset "Cluster-robust with GLM" begin
        rng = StableRNG(1234)
        n = 40
        df = DataFrame(
            Y = randn(rng, n),
            X1 = randn(rng, n),
            X2 = randn(rng, n),
            cl = repeat(1:8, inner = 5)
        )
        m = lm(@formula(Y ~ X1 + X2), df)

        # Test cluster-robust estimators
        for cr in [CR0(df.cl), CR1(df.cl)]
            V = vcov(cr, m)
            @test size(V) == (3, 3)
            @test all(diag(V) .>= 0)
        end
    end

    @testset "Linear Model - QR Decomposition" begin
        rng = StableRNG(1234)
        X = randn(rng, 50, 3)
        y = X * [1.0, -0.5, 0.5] + randn(rng, 50)

        # Construct and properly fit QR model using fit!
        pp_qr = GLM.DensePredQR(X)
        rr = GLM.LmResp(y, Vector{Float64}())
        model_qr = GLM.LinearModel(rr, pp_qr)
        GLM.fit!(model_qr)

        # Check vcov works
        V_qr = vcov(HC0(), model_qr)
        @test size(V_qr) == (3, 3)
        @test all(diag(V_qr) .>= 0)

        # Compare with Cholesky model (standard lm)
        model_chol = lm(X, y)

        # Coefficients should match
        @test coef(model_qr) ≈ coef(model_chol) atol = 1e-10

        # Residuals should match
        @test CovarianceMatrices._residuals(model_qr) ≈ CovarianceMatrices._residuals(model_chol) atol = 1e-10

        # Bread matrices should match
        @test CovarianceMatrices.bread(model_qr) ≈ CovarianceMatrices.bread(model_chol) atol = 1e-10

        # Moment matrices should match
        @test momentmatrix(model_qr) ≈ momentmatrix(model_chol) atol = 1e-10

        # HC variance estimates should match
        V_chol = vcov(HC0(), model_chol)
        @test V_qr ≈ V_chol atol = 1e-10
    end
end
