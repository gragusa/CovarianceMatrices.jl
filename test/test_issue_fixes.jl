## Regression tests for three latent issues fixed together:
##
##  1. `Uncorrelated()` had no `avar`/`residual_adjustment` methods and errored on
##     both matrices and models. It now behaves like HC0/HR0 (White's estimator).
##  2. On a bare moment matrix the finite-sample CR variants silently collapsed to
##     CR0. A dedicated `Cluster` estimator now serves the matrix/mean case, and
##     `CR1`/`CR2`/`CR3` raise an informative error on a bare matrix.
##  3. `VARHAC` and the smoothed-moment estimators (`UniformSmoother`,
##     `TriangularSmoother`) had no `residual_adjustment(::_, ::RegressionModel)`
##     method and errored when used with a fitted model.

using CovarianceMatrices, DataFrames, Test, Random, StableRNGs, Statistics,
      LinearAlgebra, GLM

const CM = CovarianceMatrices

@testset "Issue fixes (Uncorrelated / Cluster / VARHAC & smoothers on models)" begin
    n = 200
    rng = StableRNG(20240722)
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    y = 1.0 .+ 0.5 .* x1 .- 0.3 .* x2 .+ randn(rng, n)
    df = DataFrame(y = y, x1 = x1, x2 = x2)
    model = lm(@formula(y ~ x1 + x2), df)

    G = 20
    clusters = repeat(1:G, inner = n ÷ G)          # length n, 20 clusters
    clusters2 = repeat(1:4, outer = n ÷ 4)         # length n, second dimension

    # -----------------------------------------------------------------
    # 1. `Uncorrelated` behaves like HC0 / HR0 (White's estimator)
    # -----------------------------------------------------------------
    @testset "Uncorrelated == HC0/HR0" begin
        Z = randn(StableRNG(1), 120, 3)

        # Matrix interface: same meat as HC0/HR0 for every `scale` setting.
        @test aVar(Uncorrelated(), Z) ≈ aVar(HC0(), Z)
        @test aVar(Uncorrelated(), Z) ≈ aVar(HR0(), Z)
        @test aVar(Uncorrelated(), Z; scale = false) ≈ aVar(HC0(), Z; scale = false)
        # scale=false is the T-fold sum of the default (scale=true) result.
        @test aVar(Uncorrelated(), Z; scale = false) ≈ aVar(Uncorrelated(), Z) .* size(Z, 1)
        # integer scale divides by that integer (DOF-style correction).
        @test aVar(Uncorrelated(), Z; scale = size(Z, 1) - 1) ≈
              aVar(Uncorrelated(), Z; scale = false) ./ (size(Z, 1) - 1)

        # Model interface: identical to White's HC0.
        @test vcov(Uncorrelated(), model) ≈ vcov(HC0(), model)
        @test stderror(Uncorrelated(), model) ≈ stderror(HC0(), model)
    end

    # -----------------------------------------------------------------
    # 2. `Cluster` for the matrix/mean case; CR1/CR2/CR3 error on a matrix
    # -----------------------------------------------------------------
    @testset "Cluster estimator and CR-on-matrix guard" begin
        Z = randn(StableRNG(2), n, 2)

        # `Cluster` reproduces the raw cluster sum (== CR0's matrix behavior).
        @test aVar(Cluster(clusters), Z) ≈ aVar(CR0(clusters), Z)
        @test size(aVar(Cluster(clusters), Z)) == (2, 2)

        # Multi-way clustering works via the same inclusion-exclusion machinery.
        @test size(aVar(Cluster((clusters, clusters2)), Z)) == (2, 2)

        # The finite-sample variants cannot be formed from a bare matrix.
        @test_throws ArgumentError aVar(CR1(clusters), Z)
        @test_throws ArgumentError aVar(CR2(clusters), Z)
        @test_throws ArgumentError aVar(CR3(clusters), Z)

        # CR0 on a bare matrix remains available (backward compatible).
        @test size(aVar(CR0(clusters), Z)) == (2, 2)

        # `Cluster` on a fitted model errors and points to the CR family.
        @test_throws ArgumentError vcov(Cluster(clusters), model)

        # The CR family still works on a fitted model.
        @test all(isfinite, stderror(CR1(clusters), model))
        @test all(isfinite, stderror(CR2(clusters), model))
    end

    # -----------------------------------------------------------------
    # 3. VARHAC and smoothed-moment estimators work on fitted models
    # -----------------------------------------------------------------
    @testset "VARHAC & smoothers on RegressionModel" begin
        se_varhac = stderror(VARHAC(), model)
        @test length(se_varhac) == 3
        @test all(isfinite, se_varhac)
        @test all(>(0), se_varhac)

        mT = max(2, round(Int, 2.0 * n^(1 / 3)))
        se_unif = stderror(UniformSmoother(mT), model)
        @test all(isfinite, se_unif)
        @test all(>(0), se_unif)

        se_tri = stderror(TriangularSmoother(mT), model)
        @test all(isfinite, se_tri)
        @test all(>(0), se_tri)

        # And they continue to work on a bare matrix.
        Z = randn(StableRNG(3), 300, 2)
        @test size(aVar(VARHAC(), Z)) == (2, 2)
        @test size(aVar(UniformSmoother(5), Z)) == (2, 2)
    end
end
