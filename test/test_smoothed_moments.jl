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

        G̃triangular = [110/7 510/7;
                         155/7 605/7;
                         29 683/7;
                         36 106;
                         43 113;
                         50 120;
                         57 127;
                         305/7 95;
                         220/7 470/7;
                         146/7 306/7]

        k = CovarianceMatrices.UniformSmoother(m_T = m_T)
        @test k.m_T == m_T
        @test k.S_T == S_T
        k = CovarianceMatrices.UniformSmoother(S_T = S_T)
        @test k.m_T == m_T
        @test k.S_T == S_T
        G_uniform = CovarianceMatrices.smooth_moments(G, k)
        @test G_uniform≈G̃uniform rtol=1e-14
        k = CovarianceMatrices.TriangularSmoother(m_T = 3)
        G_triangular = CovarianceMatrices.smooth_moments(G, k)
        @test G_triangular≈G̃triangular rtol=1e-08

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

        G̃triangular = [215/9 865/9;
                         287/9 1007/9;
                         364/9 126;
                         148/3 1244/9;
                         175/3 445/3;
                         202/3 472/3;
                         500/9 380/3;
                         133/3 889/9;
                         305/9 665/9;
                         220/9 470/9]

        k = CovarianceMatrices.UniformSmoother(m_T = m_T)
        @test k.m_T == (2S_T -  1) / 2
        @test k.S_T == (2m_T + 1) / 2
        k = CovarianceMatrices.UniformSmoother(S_T = S_T)
        @test k.m_T == (2S_T - 1) / 2
        @test k.S_T == (2m_T + 1) / 2
        G_uniform = CovarianceMatrices.smooth_moments(G, k)
        @test G_uniform == G̃uniform

        k = CovarianceMatrices.TriangularSmoother(m_T = m_T)
        @test k.m_T == (2S_T - 1) / 2
        @test k.S_T == (2m_T + 1) / 2
        k = CovarianceMatrices.TriangularSmoother(S_T = S_T)
        @test k.m_T == (2S_T - 1) / 2
        @test k.S_T == (2m_T + 1) / 2
        G_triangular = CovarianceMatrices.smooth_moments(G, k)
        @test G_triangular ≈ G̃triangular
    end

    @testset "Kernel Functions" begin
        @testset "UniformSmoother" begin
            k = CovarianceMatrices.UniformSmoother(S_T = 2.0)

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
            k = CovarianceMatrices.TriangularSmoother(S_T = 2.0)

            # Test kernel function values
            @test CovarianceMatrices.kernel_func(k, 0.0) == 1.0
            @test CovarianceMatrices.kernel_func(k, 0.5) == 1 - 0.5 / 2.0
            @test CovarianceMatrices.kernel_func(k, 1.0) == 1 - 1.0 / 2.0
            @test CovarianceMatrices.kernel_func(k, -0.5) == 1 - 0.5 / 2.0
            @test CovarianceMatrices.kernel_func(k, -1.0) == 1 - 1.0 / 2.0
            @test CovarianceMatrices.kernel_func(k, 3.1) == 0.0
            @test CovarianceMatrices.kernel_func(k, -3.1) == 0.0

            @test CovarianceMatrices.kernel_k1(k) == 1.0
            @test CovarianceMatrices.kernel_k2(k) == 2 / 3
            @test CovarianceMatrices.kernel_k3(k) == 1 / 2
        end
    end

    @testset "Smoothing Functions" begin
        @testset "UniformSmoother" begin
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
        @testset "TriangularSmoother" begin
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
    end

    @testset "Smoothing aVar" begin
        @testset "UniformSmoother" begin
            X = randn(rng, 50000, 3)
            aVar(Bartlett(12), X)  # Warm-up
            aVar(CovarianceMatrices.TriangularSmoother(S_T=12), X)  # Warm-up
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
        @testset "TriangularSmoother" begin
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
    end

end
