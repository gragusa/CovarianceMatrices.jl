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

        G̃triangular = [4.28571 27.1429;
                        7.28571 37.2857;
                        10.7143 45.0;
                        14.2857 50.0;
                        17.8571 53.5714;
                        21.4286 57.1429;
                        25.0 60.7143;
                        27.0 61.2857;
                        25.7143 55.7143;
                        20.8571 43.7143];

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

        G̃triangular = [6.11111 33.8889;
                        9.66667 45.2222;
                        13.7778 54.8889;
                        18.2222 62.6667;
                        22.7778 68.3333;
                        27.3333 72.8889;
                        30.6667 75.1111;
                        31.4444 72.5556;
                        29.4444 65.0;
                        24.4444 52.2222]

        k = CovarianceMatrices.UniformSmoother(m_T)
        @test k.m_T == (2S_T - 1) / 2
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
            (T = 50, k = 2, m_T = 2),
            (T = 50, k = 5, m_T = 5),
            (T = 100, k = 2, m_T = 2),
            (T = 100, k = 5, m_T = 5),
            (T = 100, k = 10, m_T = 10),
            (T = 500, k = 2, m_T = 2),
            (T = 500, k = 5, m_T = 10),
            (T = 500, k = 10, m_T = 5)
        ]

        for params in test_params
            @testset "T=$(params.T), k=$(params.k), m_T=$(params.m_T)" begin
                G = randn(rng, params.T, params.k)

                # Test UniformSmoother
                k_uniform = CovarianceMatrices.UniformSmoother(params.m_T)
                G_fast_uniform = CovarianceMatrices.smooth_moments(G, k_uniform)
                G_plain_uniform = CovarianceMatrices.smooth_uniform(G, params.m_T)
                @test G_fast_uniform ≈ G_plain_uniform rtol = 1e-10

                # Test TriangularSmoother
                k_triangular = CovarianceMatrices.TriangularSmoother(params.m_T)
                G_fast_triangular = CovarianceMatrices.smooth_moments(G, k_triangular)
                G_plain_triangular = CovarianceMatrices.smooth_triangular(
                    G, params.m_T)
                @test G_fast_triangular ≈ G_plain_triangular rtol = 1e-10
            end
        end
    end

    @testset "In-place vs Out-of-place Smoothing" begin
        # Test that in-place smoothing matches out-of-place
        test_configs = [
            (T = 50, k = 3, m_T = 2),
            (T = 100, k = 5, m_T = 5),
            (T = 200, k = 10, m_T = 10)
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
            X = randn(rng, 50000, 3);
            A = aVar(Parzen(12), X)  # Warm-up
            B = aVar(CovarianceMatrices.TriangularSmoother(11), X)  # Warm-up
            @test A ≈ B atol=1e-1
            ## Test inplace and out-of-place smoothing
            ## THEY SHOULD GIVE THE SAME RESULTS EVEN WHEN RUNNED MULTIPLE TIMES
            ## Using G̃triangular from previous test for which we know the smoothing
        end
    end

    @testset "Smoothed Moments Mathematical Correctness" begin
        # These tests verify the mathematical correctness of the smoothing computation
        # by manually computing expected values following the formulas

        @testset "Uniform Smoothing - Manual Verification" begin
            # For uniform smoothing with bandwidth m_T:
            # G_smooth[t] = sum of G[i] for i in [max(1, t-m_T), min(T, t+m_T)]

            # Simple 5-element vector, m_T = 1
            G = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1)
            m_T = 1

            # Manual calculation:
            # t=1: window [max(1,0), min(5,2)] = [1,2], sum = 1+2 = 3
            # t=2: window [max(1,1), min(5,3)] = [1,3], sum = 1+2+3 = 6
            # t=3: window [max(1,2), min(5,4)] = [2,4], sum = 2+3+4 = 9
            # t=4: window [max(1,3), min(5,5)] = [3,5], sum = 3+4+5 = 12
            # t=5: window [max(1,4), min(5,6)] = [4,5], sum = 4+5 = 9
            expected = reshape([3.0, 6.0, 9.0, 12.0, 9.0], 5, 1)

            k_uniform = CovarianceMatrices.UniformSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_uniform)
            @test result ≈ expected

            # Test with m_T = 2
            m_T = 2
            # t=1: window [1, min(5,3)] = [1,3], sum = 1+2+3 = 6
            # t=2: window [1, min(5,4)] = [1,4], sum = 1+2+3+4 = 10
            # t=3: window [1, 5], sum = 1+2+3+4+5 = 15
            # t=4: window [max(1,2), 5] = [2,5], sum = 2+3+4+5 = 14
            # t=5: window [max(1,3), 5] = [3,5], sum = 3+4+5 = 12
            expected_m2 = reshape([6.0, 10.0, 15.0, 14.0, 12.0], 5, 1)

            k_uniform_2 = CovarianceMatrices.UniformSmoother(m_T)
            result_m2 = CovarianceMatrices.smooth_moments(G, k_uniform_2)
            @test result_m2 ≈ expected_m2
        end

        @testset "Triangular Smoothing - Manual Verification" begin
            # For triangular smoothing with bandwidth m_T:
            # G_smooth[t] = sum of w(i,t) * G[i] for i in window
            # where w(i,t) = 1 - 2*|t-i|/(2*m_T+1)

            # Simple 5-element vector, m_T = 1
            # scale = 2/(2*1+1) = 2/3
            G = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1)
            m_T = 1
            scale = 2.0 / (2 * m_T + 1)  # = 2/3

            # Manual calculation (using the actual implementation logic):
            # For each t, weight for index i is: 1 - scale*|t-i|
            # t=1: window [1,2]
            #   i=1: weight = 1 - (2/3)*0 = 1.0, contrib = 1.0 * 1 = 1.0
            #   i=2: weight = 1 - (2/3)*1 = 1/3, contrib = (1/3) * 2 = 2/3
            #   total = 1.0 + 2/3 = 5/3
            # t=2: window [1,3]
            #   i=1: weight = 1 - (2/3)*1 = 1/3, contrib = (1/3) * 1 = 1/3
            #   i=2: weight = 1 - (2/3)*0 = 1.0, contrib = 1.0 * 2 = 2.0
            #   i=3: weight = 1 - (2/3)*1 = 1/3, contrib = (1/3) * 3 = 1.0
            #   total = 1/3 + 2.0 + 1.0 = 10/3
            # t=3: window [2,4]
            #   i=2: weight = 1/3, contrib = (1/3) * 2 = 2/3
            #   i=3: weight = 1.0, contrib = 1.0 * 3 = 3.0
            #   i=4: weight = 1/3, contrib = (1/3) * 4 = 4/3
            #   total = 2/3 + 3.0 + 4/3 = 5.0
            # t=4: window [3,5]
            #   i=3: weight = 1/3, contrib = (1/3) * 3 = 1.0
            #   i=4: weight = 1.0, contrib = 1.0 * 4 = 4.0
            #   i=5: weight = 1/3, contrib = (1/3) * 5 = 5/3
            #   total = 1.0 + 4.0 + 5/3 = 20/3
            # t=5: window [4,5]
            #   i=4: weight = 1/3, contrib = (1/3) * 4 = 4/3
            #   i=5: weight = 1.0, contrib = 1.0 * 5 = 5.0
            #   total = 4/3 + 5.0 = 19/3
            expected = reshape([5/3, 10/3, 5.0, 20/3, 19/3], 5, 1)

            k_triangular = CovarianceMatrices.TriangularSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_triangular)
            @test result ≈ expected rtol=1e-10
        end

        @testset "Uniform Smoothing - Multi-column Matrix" begin
            # Test with a 2-column matrix to verify column-wise operation
            G = [1.0 10.0;
                 2.0 20.0;
                 3.0 30.0;
                 4.0 40.0;
                 5.0 50.0]
            m_T = 1

            # Each column should be smoothed independently
            # Column 1: [3, 6, 9, 12, 9] (from previous test)
            # Column 2: [30, 60, 90, 120, 90] (10x column 1)
            expected = [3.0 30.0;
                        6.0 60.0;
                        9.0 90.0;
                        12.0 120.0;
                        9.0 90.0]

            k_uniform = CovarianceMatrices.UniformSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_uniform)
            @test result ≈ expected
        end

        @testset "Triangular Smoothing - Multi-column Matrix" begin
            # Test with a 2-column matrix
            G = [1.0 10.0;
                 2.0 20.0;
                 3.0 30.0;
                 4.0 40.0;
                 5.0 50.0]
            m_T = 1

            # Each column should be smoothed independently
            # Column 2 is 10x column 1, so result should be 10x as well
            expected_col1 = [5/3, 10/3, 5.0, 20/3, 19/3]
            expected = hcat(expected_col1, 10.0 .* expected_col1)

            k_triangular = CovarianceMatrices.TriangularSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_triangular)
            @test result ≈ expected rtol=1e-10
        end

        @testset "Smoothing Preserves Sum (Uniform)" begin
            # For uniform smoothing, the sum of all smoothed values should equal
            # the sum of original values times the average window size
            # This is a useful sanity check
            G = randn(StableRNG(42), 20, 3)
            m_T = 3

            k_uniform = CovarianceMatrices.UniformSmoother(m_T)
            G_smooth = CovarianceMatrices.smooth_moments(G, k_uniform)

            # Each element contributes to multiple windows
            # Interior elements (with full windows) contribute to 2*m_T+1 sums
            # Boundary elements contribute to fewer
            # Verify the computation is consistent
            for j in 1:size(G, 2)
                # Manually compute expected sum
                T = size(G, 1)
                expected_sum = 0.0
                for t in 1:T
                    a = max(1, t - m_T)
                    b = min(T, t + m_T)
                    expected_sum += sum(G[a:b, j])
                end
                @test sum(G_smooth[:, j]) ≈ expected_sum rtol=1e-10
            end
        end

        @testset "Smoothing with m_T = 0 Returns Original" begin
            # With m_T = 0, window is just the element itself
            G = randn(StableRNG(123), 10, 4)

            k_uniform_0 = CovarianceMatrices.UniformSmoother(0)
            k_triangular_0 = CovarianceMatrices.TriangularSmoother(0)

            G_uniform_0 = CovarianceMatrices.smooth_moments(G, k_uniform_0)
            G_triangular_0 = CovarianceMatrices.smooth_moments(G, k_triangular_0)

            @test G_uniform_0 ≈ G
            @test G_triangular_0 ≈ G
        end

        @testset "Boundary Behavior Verification" begin
            # Verify boundary elements are handled correctly
            G = reshape(collect(1.0:10.0), 10, 1)
            m_T = 5  # Large bandwidth relative to data

            k_uniform = CovarianceMatrices.UniformSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_uniform)

            # t=1: window [1, min(10, 6)] = [1,6], sum = 1+2+3+4+5+6 = 21
            @test result[1] ≈ 21.0

            # t=5: window [max(1,0), min(10,10)] = [1,10], sum = 55
            @test result[5] ≈ 55.0

            # t=6: window [max(1,1), min(10,11)] = [1,10], sum = 55
            @test result[6] ≈ 55.0

            # t=10: window [max(1,5), 10] = [5,10], sum = 5+6+7+8+9+10 = 45
            @test result[10] ≈ 45.0
        end

        @testset "Triangular Weights Sum to Expected Value" begin
            # For triangular kernel, verify the weights computation
            m_T = 2
            scale = 2.0 / (2 * m_T + 1)  # = 2/5 = 0.4

            # With all 1s input, output should equal sum of weights in window
            G = ones(7, 1)

            k_triangular = CovarianceMatrices.TriangularSmoother(m_T)
            result = CovarianceMatrices.smooth_moments(G, k_triangular)

            # t=3 (middle, full window [1,5]):
            # weights: 1-0.4*2, 1-0.4*1, 1-0.4*0, 1-0.4*1, 1-0.4*2
            #        = 0.2, 0.6, 1.0, 0.6, 0.2
            # sum = 2.6
            @test result[3] ≈ 2.6 rtol=1e-10

            # t=4 (middle, full window [2,6]):
            # Same weights, sum = 2.6
            @test result[4] ≈ 2.6 rtol=1e-10

            # t=1 (boundary, window [1,3]):
            # weights: 1.0, 0.6, 0.2
            # sum = 1.8
            @test result[1] ≈ 1.8 rtol=1e-10
        end

        @testset "Verify Against Naive Loop Implementation" begin
            # Compare against a simple naive loop implementation
            function naive_uniform_smooth(G, m_T)
                T, k = size(G)
                result = zeros(T, k)
                for t in 1:T
                    for j in 1:k
                        for i in max(1, t - m_T):min(T, t + m_T)
                            result[t, j] += G[i, j]
                        end
                    end
                end
                return result
            end

            function naive_triangular_smooth(G, m_T)
                T, k = size(G)
                scale = 2.0 / (2 * m_T + 1)
                result = zeros(T, k)
                for t in 1:T
                    for j in 1:k
                        for i in max(1, t - m_T):min(T, t + m_T)
                            weight = 1.0 - scale * abs(t - i)
                            result[t, j] += weight * G[i, j]
                        end
                    end
                end
                return result
            end

            # Test with various configurations
            for (T, k, m_T) in [(10, 2, 2), (20, 3, 5), (50, 1, 10), (100, 4, 3)]
                G = randn(StableRNG(T * k * m_T), T, k)

                k_uniform = CovarianceMatrices.UniformSmoother(m_T)
                k_triangular = CovarianceMatrices.TriangularSmoother(m_T)

                result_uniform = CovarianceMatrices.smooth_moments(G, k_uniform)
                result_triangular = CovarianceMatrices.smooth_moments(G, k_triangular)

                expected_uniform = naive_uniform_smooth(G, m_T)
                expected_triangular = naive_triangular_smooth(G, m_T)

                @test result_uniform ≈ expected_uniform rtol=1e-10
                @test result_triangular ≈ expected_triangular rtol=1e-10
            end
        end
    end

    @testset "Integer Input Dispatch" begin
        # Test that Int matrices are properly converted to Float and processed correctly
        # These tests cover the Int dispatch wrappers: uniform_sum!, uniform_sum,
        # triangular_sum!, triangular_sum with Int input

        @testset "uniform_sum with Int input" begin
            # Create an Int matrix
            G_int = [1 11; 2 12; 3 13; 4 14; 5 15; 6 16; 7 17; 8 18; 9 19; 10 20]
            G_float = Float64.(G_int)
            m_T = 3

            # Test out-of-place version
            result_int = CovarianceMatrices.uniform_sum(G_int, m_T)
            result_float = CovarianceMatrices.uniform_sum(G_float, m_T)
            @test result_int ≈ result_float
            @test eltype(result_int) <: AbstractFloat

            # Test in-place version
            dest_from_int = zeros(Float64, size(G_int))
            dest_from_float = zeros(Float64, size(G_float))
            CovarianceMatrices.uniform_sum!(dest_from_int, G_int, m_T)
            CovarianceMatrices.uniform_sum!(dest_from_float, G_float, m_T)
            @test dest_from_int ≈ dest_from_float
        end

        @testset "triangular_sum with Int input" begin
            # Create an Int matrix
            G_int = [1 11; 2 12; 3 13; 4 14; 5 15; 6 16; 7 17; 8 18; 9 19; 10 20]
            G_float = Float64.(G_int)
            m_T = 3

            # Test out-of-place version
            result_int = CovarianceMatrices.triangular_sum(G_int, m_T)
            result_float = CovarianceMatrices.triangular_sum(G_float, m_T)
            @test result_int ≈ result_float
            @test eltype(result_int) <: AbstractFloat

            # Test in-place version
            dest_from_int = zeros(Float64, size(G_int))
            dest_from_float = zeros(Float64, size(G_float))
            CovarianceMatrices.triangular_sum!(dest_from_int, G_int, m_T)
            CovarianceMatrices.triangular_sum!(dest_from_float, G_float, m_T)
            @test dest_from_int ≈ dest_from_float
        end

        @testset "smooth_moments with Int input via smoother API" begin
            G_int = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15]
            G_float = Float64.(G_int)
            m_T = 2

            # Test UniformSmoother with Int input
            k_uniform = CovarianceMatrices.UniformSmoother(m_T)
            result_uniform_int = CovarianceMatrices.smooth_moments(G_int, k_uniform)
            result_uniform_float = CovarianceMatrices.smooth_moments(G_float, k_uniform)
            @test result_uniform_int ≈ result_uniform_float

            # Test TriangularSmoother with Int input
            k_triangular = CovarianceMatrices.TriangularSmoother(m_T)
            result_triangular_int = CovarianceMatrices.smooth_moments(G_int, k_triangular)
            result_triangular_float = CovarianceMatrices.smooth_moments(G_float, k_triangular)
            @test result_triangular_int ≈ result_triangular_float
        end
    end

    @testset "Kernel Statistics Functions" begin
        # Test k1hat, k2hat, k3hat for correctness

        @testset "k2hat" begin
            # k2hat is used in avar, verify it computes sum of squared kernel values
            k_uniform = CovarianceMatrices.UniformSmoother(2)
            k_triangular = CovarianceMatrices.TriangularSmoother(2)

            # For uniform kernel with m_T=2: kernel is 1 for |s| <= 2, so k2hat = sum of 1^2 for s=-2,-1,0,1,2 = 5
            @test CovarianceMatrices.k2hat(k_uniform) == 5.0

            # For triangular kernel: sum of (1 - |s|/S_T)^2 for s in [-m_T, m_T]
            # S_T = (2*2+1)/2 = 2.5
            # s=-2: (1 - 2/2.5)^2 = (0.2)^2 = 0.04
            # s=-1: (1 - 1/2.5)^2 = (0.6)^2 = 0.36
            # s=0:  (1 - 0)^2 = 1.0
            # s=1:  (1 - 1/2.5)^2 = (0.6)^2 = 0.36
            # s=2:  (1 - 2/2.5)^2 = (0.2)^2 = 0.04
            # Total = 0.04 + 0.36 + 1.0 + 0.36 + 0.04 = 1.8
            @test CovarianceMatrices.k2hat(k_triangular) ≈ 1.8
        end

        @testset "k1hat" begin
            # k1hat returns kernel function values over extended range [-m_T-2, m_T+2]
            k_uniform = CovarianceMatrices.UniformSmoother(2)
            k_triangular = CovarianceMatrices.TriangularSmoother(2)

            k1_uniform = CovarianceMatrices.k1hat(k_uniform)
            k1_triangular = CovarianceMatrices.k1hat(k_triangular)

            # For uniform with m_T=2, S_T=2.5, kernel is 1 for |s/S_T| <= 1
            # Range is s in [-4, 4], so 9 values
            @test length(k1_uniform) == 9  # -4,-3,-2,-1,0,1,2,3,4

            # Kernel is 1 for |s| <= S_T = 2.5, i.e., s in {-2,-1,0,1,2}
            # Expected: [0,0,1,1,1,1,1,0,0] for s=-4,-3,-2,-1,0,1,2,3,4
            @test k1_uniform == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

            # For triangular, similar range check
            @test length(k1_triangular) == 9
            # Kernel is (1 - |s|/S_T) for |s| <= S_T, 0 otherwise
            # At s=0: 1.0, s=±1: 0.6, s=±2: 0.2, s=±3,±4: 0.0
            @test k1_triangular[5] ≈ 1.0  # s=0
            @test k1_triangular[4] ≈ 0.6  # s=-1
            @test k1_triangular[6] ≈ 0.6  # s=1
            @test k1_triangular[3] ≈ 0.2  # s=-2
            @test k1_triangular[7] ≈ 0.2  # s=2
            @test k1_triangular[1] == 0.0  # s=-4
            @test k1_triangular[9] == 0.0  # s=4
        end

        @testset "k3hat" begin
            # k3hat computes sum of cubed kernel values over extended range
            k_uniform = CovarianceMatrices.UniformSmoother(2)
            k_triangular = CovarianceMatrices.TriangularSmoother(2)

            # For uniform kernel, k(s)^3 = 1 where kernel is 1, so same as counting
            # s in [-4,4], kernel is 1 for s in [-2,2], so 5 values
            @test CovarianceMatrices.k3hat(k_uniform) == 5.0

            # For triangular: sum of (1 - |s|/S_T)^3 for s in [-4, 4] where kernel > 0
            # Only s in [-2,2] contribute since |s|/S_T <= 1 means |s| <= 2.5
            # s=-2: 0.2^3 = 0.008
            # s=-1: 0.6^3 = 0.216
            # s=0:  1.0^3 = 1.0
            # s=1:  0.6^3 = 0.216
            # s=2:  0.2^3 = 0.008
            # Total = 0.008 + 0.216 + 1.0 + 0.216 + 0.008 = 1.448
            @test CovarianceMatrices.k3hat(k_triangular) ≈ 1.448
        end
    end

    @testset "Optimal Bandwidth Functions" begin
        # Test optimal_bandwidth for both kernel types

        @testset "UniformSmoother optimal_bandwidth" begin
            # Formula: 2.0 * T^(1/3)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.UniformSmoother(0), 100) ≈
                  2.0 * 100^(1/3)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.UniformSmoother(0), 1000) ≈
                  2.0 * 1000^(1/3)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.UniformSmoother(5), 500) ≈
                  2.0 * 500^(1/3)
        end

        @testset "TriangularSmoother optimal_bandwidth" begin
            # Formula: 1.5 * T^(1/5)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.TriangularSmoother(0), 100) ≈
                  1.5 * 100^(1/5)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.TriangularSmoother(0), 1000) ≈
                  1.5 * 1000^(1/5)
            @test CovarianceMatrices.optimal_bandwidth(CovarianceMatrices.TriangularSmoother(5), 500) ≈
                  1.5 * 500^(1/5)
        end
    end
end
