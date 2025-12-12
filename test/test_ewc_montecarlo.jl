## Monte Carlo validation tests for EWC variance estimator
##
## This file tests that the EWC estimator produces correct coverage rates
## when used to construct confidence intervals for linear regression parameters.

using CovarianceMatrices
using Test
using Random
using LinearAlgebra
using Statistics
using GLM
using DataFrames

@testset "EWC Monte Carlo Coverage Tests" begin
    @testset "EWC coverage with AR(1) errors" begin
        """
        Monte Carlo simulation to verify EWC produces ~95% coverage.

        DGP: y = β₀ + β₁*x + ε
        where ε follows AR(1): ε_t = ρ*ε_{t-1} + u_t, u_t ~ N(0,1)

        We estimate OLS, compute EWC robust standard errors, and check
        that the 95% CI covers the true β₁ approximately 95% of the time.
        """
        function monte_carlo_coverage(;
                n_sims::Int = 1000,
                n_obs::Int = 200,
                B::Int = 10,
                rho::Float64 = 0.5,
                seed::Int = 12345)
            rng = Random.Xoshiro(seed)
            coverage_count = 0
            true_beta = 2.0  # slope coefficient

            for _ in 1:n_sims
                x = randn(rng, n_obs)

                # Generate AR(1) errors for autocorrelation
                e = Vector{Float64}(undef, n_obs)
                e[1] = randn(rng)
                for t in 2:n_obs
                    e[t] = rho * e[t - 1] + randn(rng)
                end
                y = 1.0 .+ true_beta .* x .+ e

                # Fit OLS model
                df = DataFrame(y = y, x = x)
                model = lm(@formula(y ~ x), df)

                # EWC robust variance using vcov interface
                V = vcov(EWC(B), model)

                # 95% CI for slope (coefficient index 2)
                beta_hat = coef(model)[2]
                se = sqrt(V[2, 2])
                lower = beta_hat - 1.96 * se
                upper = beta_hat + 1.96 * se

                if lower <= true_beta <= upper
                    coverage_count += 1
                end
            end

            return coverage_count / n_sims
        end

        # Run Monte Carlo with moderate sample size
        coverage = monte_carlo_coverage(n_sims = 1000, n_obs = 200, B = 10)

        # Coverage should be approximately 95%, allow for Monte Carlo error
        # With 1000 simulations, SE ≈ sqrt(0.95*0.05/1000) ≈ 0.007
        # So 95% CI for coverage is approximately [0.93, 0.97]
        @test 0.90 <= coverage <= 0.99
        println("  EWC coverage rate: $(round(coverage * 100, digits=1))%")
    end

    @testset "EWC vs HAC comparison" begin
        """
        Verify that EWC variance estimates are in the same ballpark as HAC.
        Both estimators should produce similar results for autocorrelated data.
        """
        rng = Random.Xoshiro(54321)
        n_obs = 200

        x = randn(rng, n_obs)
        e = Vector{Float64}(undef, n_obs)
        e[1] = randn(rng)
        for t in 2:n_obs
            e[t] = 0.5 * e[t - 1] + randn(rng)
        end
        y = 1.0 .+ 2.0 .* x .+ e

        df = DataFrame(y = y, x = x)
        model = lm(@formula(y ~ x), df)

        # Get variances from both estimators
        V_ewc = vcov(EWC(10), model)
        V_hac = vcov(Bartlett{NeweyWest}(), model)

        # Standard errors should be of similar magnitude (within factor of 3)
        se_ewc = sqrt(V_ewc[2, 2])
        se_hac = sqrt(V_hac[2, 2])

        @test 0.33 < se_ewc / se_hac < 3.0
        println("  EWC SE: $(round(se_ewc, digits=4)), HAC SE: $(round(se_hac, digits=4))")
    end

    @testset "EWC varying B parameter" begin
        """
        Test that EWC works correctly with different numbers of basis functions.
        """
        rng = Random.Xoshiro(99999)
        n_obs = 150

        x = randn(rng, n_obs)
        e = randn(rng, n_obs)
        y = 1.0 .+ 2.0 .* x .+ e

        df = DataFrame(y = y, x = x)
        model = lm(@formula(y ~ x), df)

        # Test with different B values
        for B in [3, 5, 10, 20]
            V = vcov(EWC(B), model)
            @test size(V) == (2, 2)
            # Check approximate symmetry (floating-point precision)
            @test maximum(abs.(V - V')) < 1e-14
            # Check positive definiteness using Symmetric wrapper
            @test isposdef(Symmetric(V))
        end
    end
end
