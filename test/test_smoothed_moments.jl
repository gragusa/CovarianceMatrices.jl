"""
Tests for Smith's Smoothed Moments Variance Estimation

This test suite validates the SmoothedMoments estimator implementation against:
1. Basic functionality and edge cases
2. Comparison with HAC estimators for large samples (asymptotic equivalence)
3. Different kernel types and bandwidth settings
4. Performance and type stability
"""

using Test
using CovarianceMatrices
using LinearAlgebra
using Statistics
using StableRNGs
using Random

@testset "Smoothed Moments Tests" begin
    # Set up stable RNG for reproducible tests
    rng = StableRNG(123456)

    @testset "Correctness" begin
        # Test m_T = 3 -> bw = (2*m_T + 1)
        T = 10
        m_T = 3
        S_T = (2 * m_T + 1) / 2
        G = hcat(collect(1.0:10), collect(11.0:20))  # 10×2 matrix
        G̃uniform = [10 50;
                     15 65;
                     21 81;
                     28 98;
                     35 105;
                     42 112;
                     49 119;
                     45 105;
                     40 90;
                     34 74]

        G̃triangular = [ 4.28571  27.1429;
                        7.28571  37.2857;
                       10.7143   45.0;
                       14.2857   50.0;
                       17.8571   53.5714;
                       21.4286   57.1429;
                       25.0      60.7143;
                       27.0      61.2857;
                       25.7143   55.7143;
                       20.8571   43.7143];

        k = CovarianceMatrices.UniformSmoother(m_T)
        @test k.m_T == m_T
        @test CovarianceMatrices.S_T(k) == (2*m_T + 1) / 2
        G_uniform = CovarianceMatrices.smooth_moments(G, k)
        @test G_uniform≈G̃uniform rtol=1e-14
        k = CovarianceMatrices.TriangularSmoother(3)
        G_triangular = CovarianceMatrices.smooth_moments(G, k)
        @test G_triangular≈G̃triangular rtol=1e-05

        m_T = 4
        S_T = (2m_T + 1) / 2
        G̃uniform = [15 65;
                     21 81;
                     28 98;
                     36 116;
                     45 135;
                     54 144;
                     52 132;
                     49 119;
                     45 105;
                     40 90]

        G̃triangular = [ 6.11111  33.8889;
                        9.66667  45.2222;
                       13.7778   54.8889;
                       18.2222   62.6667;
                       22.7778   68.3333;
                       27.3333   72.8889;
                       30.6667   75.1111;
                       31.4444   72.5556;
                       29.4444   65.0;
                       24.4444   52.2222]

        k = CovarianceMatrices.UniformSmoother(m_T)
        @test k.m_T == (2S_T -  1) / 2
        @test CovarianceMatrices.S_T(k) == (2m_T + 1) / 2
        G_uniform = CovarianceMatrices.smooth_moments(G, k)
        @test G_uniform == G̃uniform

        k = CovarianceMatrices.TriangularSmoother(m_T)
        @test k.m_T == (2S_T - 1) / 2
        @test CovarianceMatrices.S_T(k) == (2m_T + 1) / 2
        G_triangular = CovarianceMatrices.smooth_moments(G, k)
        @test G_triangular ≈ G̃triangular rtol=1e-05
    end

    @testset "Bandwidth Validation" begin
        @testset "UniformSmoother" begin
            # Test that non-integer m_T throws ArgumentError
            @test_throws ArgumentError CovarianceMatrices.UniformSmoother(2.5)
            @test_throws ArgumentError CovarianceMatrices.UniformSmoother(3.7)
            @test_throws ArgumentError CovarianceMatrices.UniformSmoother(-1)
            @test_throws ArgumentError CovarianceMatrices.UniformSmoother(-2.5)

            # Test that integer m_T works
            @test CovarianceMatrices.UniformSmoother(2).m_T == 2
            @test CovarianceMatrices.UniformSmoother(5).m_T == 5
            @test CovarianceMatrices.UniformSmoother(0).m_T == 0
        end

        @testset "TriangularSmoother" begin
            # Test that non-integer m_T throws ArgumentError
            @test_throws ArgumentError CovarianceMatrices.TriangularSmoother(2.5)
            @test_throws ArgumentError CovarianceMatrices.TriangularSmoother(3.7)
            @test_throws ArgumentError CovarianceMatrices.TriangularSmoother(-1)
            @test_throws ArgumentError CovarianceMatrices.TriangularSmoother(-2.5)

            # Test that integer m_T works
            @test CovarianceMatrices.TriangularSmoother(2).m_T == 2
            @test CovarianceMatrices.TriangularSmoother(5).m_T == 5
            @test CovarianceMatrices.TriangularSmoother(0).m_T == 0
        end
    end

    @testset "Kernel Functions" begin
        @testset "UniformSmoother" begin
            k = CovarianceMatrices.UniformSmoother(2)  # S_T = (2*2 + 1)/2 = 2.5

            # Test kernel function values
            @test CovarianceMatrices.kernel_func(k, 0.0) == 1.0
            @test CovarianceMatrices.kernel_func(k, 0.5) == 1.0
            @test CovarianceMatrices.kernel_func(k, 2.0) == 1.0
            @test CovarianceMatrices.kernel_func(k, -2.0) == 1.0
            @test CovarianceMatrices.kernel_func(k, 3.1) == 0.0
            @test CovarianceMatrices.kernel_func(k, -3.1) == 0.0

            # Test kernel constant k₂
            @test CovarianceMatrices.kernel_k1(k) == 2.0
            @test CovarianceMatrices.kernel_k2(k) == 2.0
            @test CovarianceMatrices.kernel_k3(k) == 2.0
        end

        @testset "TriangularSmoother" begin
            k = CovarianceMatrices.TriangularSmoother(2)  # S_T = (2*2 + 1)/2 = 2.5

            # Test kernel function values
            @test CovarianceMatrices.kernel_func(k, 0.0) == 1.0
            @test CovarianceMatrices.kernel_func(k, 0.5) ≈ 1 - 0.5 / 2.5
            @test CovarianceMatrices.kernel_func(k, 1.0) ≈ 1 - 1.0 / 2.5
            @test CovarianceMatrices.kernel_func(k, -0.5) ≈ 1 - 0.5 / 2.5
            @test CovarianceMatrices.kernel_func(k, -1.0) ≈ 1 - 1.0 / 2.5
            @test CovarianceMatrices.kernel_func(k, 3.1) == 0.0
            @test CovarianceMatrices.kernel_func(k, -3.1) == 0.0

            @test CovarianceMatrices.kernel_k1(k) == 1.0
            @test CovarianceMatrices.kernel_k2(k) == 2 / 3
            @test CovarianceMatrices.kernel_k3(k) == 1 / 2
        end
    end

    @testset "Edge Cases" begin
        @testset "Very Small Matrices" begin
            # T=2, k=1, m_T=1
            G_small = reshape([1.0; 2.0], 2, 1)  # Make it a 2×1 matrix
            k_uniform = CovarianceMatrices.UniformSmoother(1)
            k_triangular = CovarianceMatrices.TriangularSmoother(1)

            G_smooth_u = CovarianceMatrices.smooth_moments(G_small, k_uniform)
            G_smooth_t = CovarianceMatrices.smooth_moments(G_small, k_triangular)
            @test size(G_smooth_u) == size(G_small)
            @test size(G_smooth_t) == size(G_small)
            @test all(isfinite, G_smooth_u)
            @test all(isfinite, G_smooth_t)

            # T=5, k=2
            G_small2 = randn(rng, 5, 2)
            G_smooth_u2 = CovarianceMatrices.smooth_moments(G_small2, k_uniform)
            G_smooth_t2 = CovarianceMatrices.smooth_moments(G_small2, k_triangular)
            @test size(G_smooth_u2) == size(G_small2)
            @test size(G_smooth_t2) == size(G_small2)
        end

        @testset "Large Matrices" begin
            # T=10000, k=5, m_T=5
            G_large = randn(rng, 10000, 5)
            k_uniform = CovarianceMatrices.UniformSmoother(5)
            k_triangular = CovarianceMatrices.TriangularSmoother(5)

            G_smooth_u = CovarianceMatrices.smooth_moments(G_large, k_uniform)
            G_smooth_t = CovarianceMatrices.smooth_moments(G_large, k_triangular)
            @test size(G_smooth_u) == size(G_large)
            @test size(G_smooth_t) == size(G_large)
            @test all(isfinite, G_smooth_u)
            @test all(isfinite, G_smooth_t)
        end

        @testset "Edge Bandwidths" begin
            G_test = randn(rng, 100, 3)

            # m_T = 0 (no smoothing, should give zeros or identity-like behavior)
            k_uniform_0 = CovarianceMatrices.UniformSmoother(0)
            k_triangular_0 = CovarianceMatrices.TriangularSmoother(0)
            G_smooth_u0 = CovarianceMatrices.smooth_moments(G_test, k_uniform_0)
            G_smooth_t0 = CovarianceMatrices.smooth_moments(G_test, k_triangular_0)
            @test size(G_smooth_u0) == size(G_test)
            @test size(G_smooth_t0) == size(G_test)
            @test G_smooth_u0 ≈ G_test  # No smoothing
            @test G_smooth_t0 ≈ G_test  # No smoothing
            # m_T >> T (very large bandwidth)
            k_uniform_large = CovarianceMatrices.UniformSmoother(200)
            k_triangular_large = CovarianceMatrices.TriangularSmoother(200)
            G_smooth_u_large = CovarianceMatrices.smooth_moments(G_test, k_uniform_large)
            G_smooth_t_large = CovarianceMatrices.smooth_moments(G_test, k_triangular_large)
            @test size(G_smooth_u_large) == size(G_test)
            @test size(G_smooth_t_large) == size(G_test)
            @test all(isfinite, G_smooth_u_large)
            @test all(isfinite, G_smooth_t_large)
        end
    end

    @testset "Correctness: Fast vs Plain Implementations" begin
        # Test that fast implementations match plain reference implementations
        test_params = [
            (T=50, k=2, m_T=2),
            (T=50, k=5, m_T=5),
            (T=100, k=2, m_T=2),
            (T=100, k=5, m_T=5),
            (T=100, k=10, m_T=10),
            (T=500, k=2, m_T=2),
            (T=500, k=5, m_T=10),
            (T=500, k=10, m_T=5)
        ]

        for params in test_params
            @testset "T=$(params.T), k=$(params.k), m_T=$(params.m_T)" begin
                G = randn(rng, params.T, params.k)

                # Test UniformSmoother
                k_uniform = CovarianceMatrices.UniformSmoother(params.m_T)
                G_fast_uniform = CovarianceMatrices.smooth_moments(G, k_uniform)
                G_plain_uniform = CovarianceMatrices.smooth_uniform_plain2(G, params.m_T)
                @test G_fast_uniform ≈ G_plain_uniform rtol = 1e-10

                # Test TriangularSmoother
                k_triangular = CovarianceMatrices.TriangularSmoother(params.m_T)
                G_fast_triangular = CovarianceMatrices.smooth_moments(G, k_triangular)
                G_plain_triangular = CovarianceMatrices.smooth_triangular_plain2(
                    G, params.m_T)
                @test G_fast_triangular ≈ G_plain_triangular rtol = 1e-10
            end
        end
    end

    @testset "In-place vs Out-of-place Smoothing" begin
        # Test that in-place smoothing matches out-of-place
        test_configs = [
            (T=50, k=3, m_T=2),
            (T=100, k=5, m_T=5),
            (T=200, k=10, m_T=10)
        ]

        for config in test_configs
            @testset "T=$(config.T), k=$(config.k), m_T=$(config.m_T)" begin
                G = randn(rng, config.T, config.k)

                # UniformSmoother
                k_uniform = CovarianceMatrices.UniformSmoother(config.m_T)
                G_out = CovarianceMatrices.smooth_moments(G, k_uniform)
                G_in = similar(G)
                CovarianceMatrices.smooth_moments!(G_in, G, k_uniform)
                @test G_out ≈ G_in

                # Test that running multiple times gives same result
                G_in2 = similar(G)
                CovarianceMatrices.smooth_moments!(G_in2, G, k_uniform)
                @test G_in ≈ G_in2

                # TriangularSmoother
                k_triangular = CovarianceMatrices.TriangularSmoother(config.m_T)
                G_out_t = CovarianceMatrices.smooth_moments(G, k_triangular)
                G_in_t = similar(G)
                CovarianceMatrices.smooth_moments!(G_in_t, G, k_triangular)
                @test G_out_t ≈ G_in_t

                # Test that running multiple times gives same result
                G_in2_t = similar(G)
                CovarianceMatrices.smooth_moments!(G_in2_t, G, k_triangular)
                @test G_in_t ≈ G_in2_t
            end
        end
    end

    @testset "Smoothing aVar" begin
        @testset "UniformSmoother" begin
            X = randn(rng, 50000, 3);
            A = aVar(Bartlett(12), X)  # Warm-up
            B = aVar(CovarianceMatrices.UniformSmoother(11), X)  # Warm-up
            @test A ≈ B atol=1e-1
        end
        @testset "TriangularSmoother" begin
            A = aVar(Parzen(12), X)  # Warm-up
            B = aVar(CovarianceMatrices.TriangularSmoother(11), X)  # Warm-up
            @test A ≈ B atol=1e-1
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
    end

end
