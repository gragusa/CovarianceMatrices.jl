"""
Tests for debug=true functionality and tolerance control in matrix inversions.

This test file verifies:
1. Debug output for matrices with small singular values
2. Tolerance control with rcond_tol and rtol parameters
3. Comprehensive reporting of matrix inversion issues
4. Proper handling of near-singular matrices
"""

using CovarianceMatrices
using LinearAlgebra
using Test
using Random

@testset "Debug Inversions and Tolerance Control ✅" begin
    @testset "Matrix with Small Singular Values" begin
        # Create a matrix with one very small singular value
        Random.seed!(42)

        # Create a 3x3 matrix with controlled singular values
        U, _, V = svd(randn(3, 3))
        svals = [2.0, 1.0, 1e-10]  # One very small singular value
        A = U * Diagonal(svals) * V'

        @test cond(A) > 1e8  # Verify it's ill-conditioned

        # Test ipinv directly to get diagnostic info
        inv_A, flag, returned_svals = CovarianceMatrices.ipinv(A; rtol = 1e-8)

        @test length(flag) == 3
        @test length(returned_svals) == 3
        @test sum(flag) == 1  # One problematic singular value
        @test minimum(returned_svals) ≈ 1e-10 rtol=1e-6
        @test maximum(returned_svals) ≈ 2.0 rtol=1e-6
    end

    @testset "Debug Output for vcov" begin
        # Create test data with near-singular moment matrix
        Random.seed!(123)
        n, k = 50, 3

        # Create design matrix with severe multicollinearity
        X = randn(n, k)
        X[:, 3] = X[:, 1] + 1e-12 * randn(n)  # Nearly identical columns

        moment_matrix = X
        score_matrix = X' * X  # This will be near-singular

        @test cond(score_matrix) > 1e10  # Verify ill-conditioning

        # Capture output from debug mode
        # Note: In practice this would print to stdout,
        # here we just verify the computation succeeds

        # Test Information form with debug
        V_info = vcov(HC0(), Information(), moment_matrix;
            score = score_matrix,
            debug = true,
            cond_atol = 1e-10
        )

        @test size(V_info) == (k, k)
        @test issymmetric(V_info)
        @test isposdef(Symmetric(V_info)) || isposdef(Symmetric(V_info + 1e-10*I))  # May need regularization

        # Test Misspecified form with debug
        V_mis = vcov(HC0(), Misspecified(), moment_matrix;
            score = score_matrix,
            debug = true,
            cond_rtol = 1e-8
        )

        @test size(V_mis) == (k, k)
        @test issymmetric(V_mis)
    end

    @testset "Tolerance Parameter Control" begin
        # Create a diagonal matrix with known small eigenvalue
        A = Diagonal([2.0, 1.0, 1e-8])
        moment_matrix = randn(10, 3)

        # Test with default tolerances (nothing)
        V1 = vcov(HC0(), Information(), moment_matrix;
            score = A,
            cond_atol = nothing,
            cond_rtol = nothing
        )
        @test size(V1) == (3, 3)

        # Test with custom absolute tolerance
        V2 = vcov(HC0(), Information(), moment_matrix;
            score = A,
            cond_atol = 1e-6
        )
        @test size(V2) == (3, 3)

        # Test with custom relative tolerance
        V3 = vcov(HC0(), Information(), moment_matrix;
            score = A,
            cond_rtol = 1e-5
        )
        @test size(V3) == (3, 3)

        # All computations should succeed with different tolerance settings
        # Results may be the same if tolerances don't affect the specific matrix
        @test isfinite(sum(V1))
        @test isfinite(sum(V2))
        @test isfinite(sum(V3))
    end

    @testset "Debug with Multiple Matrix Inversions" begin
        # Test GMM case where multiple matrices need inversion
        Random.seed!(456)
        n, m, k = 100, 5, 3  # Overidentified case

        # Create moment matrix
        Z = randn(n, m)
        # Create Jacobian with some near-linear dependence
        G = randn(m, k)
        G[end, :] = G[1, :] + 1e-9 * randn(k)  # Near dependence

        # Create objective Hessian
        H = G' * G + 1e-8 * I(k)  # Regularized for numerical stability

        # Test GMM misspecified form which inverts multiple matrices
        V_gmm = vcov(HR0(), Misspecified(), Z;
            score = G,
            hessian_objective = H,
            debug = true,
            cond_atol = 1e-12
        )

        @test size(V_gmm) == (k, k)
        @test issymmetric(V_gmm)
    end

    @testset "Error Conditions with Debug" begin
        # Test that debug mode works even when matrices are very problematic
        problematic_matrix = [1.0 1.0; 1.0 1.0]  # Rank deficient
        moment_matrix = randn(20, 2)

        # This should complete despite the problematic matrix
        # (with regularization via pseudo-inverse)
        V_problem = vcov(HC0(), Information(), moment_matrix;
            score = problematic_matrix,
            debug = true,
            cond_atol = 1e-6
        )

        @test size(V_problem) == (2, 2)
        # Result may have very high condition number but should be computable
    end

    @testset "ipinv Return Values" begin
        # Test that ipinv returns all expected values correctly

        # Test with regular matrix
        A_regular = I(3) + 0.1 * randn(3, 3)
        inv_reg, flag_reg, svals_reg = CovarianceMatrices.ipinv(A_regular)

        @test size(inv_reg) == (3, 3)
        @test length(flag_reg) == 3
        @test length(svals_reg) == 3
        @test all(.!flag_reg)  # No problematic values
        @test all(svals_reg .> 0)  # All positive singular values

        # Test with diagonal matrix having small eigenvalue
        A_diag = Diagonal([2.0, 1.0, 1e-12])
        inv_diag, flag_diag, svals_diag = CovarianceMatrices.ipinv(A_diag; rtol = 1e-10)

        @test size(inv_diag) == (3, 3)
        @test length(flag_diag) == 3
        @test length(svals_diag) == 3
        @test sum(flag_diag) == 1  # One problematic value
        @test svals_diag ≈ [2.0, 1.0, 1e-12] || svals_diag ≈ [1e-12, 1.0, 2.0] # Order may vary
    end
end
