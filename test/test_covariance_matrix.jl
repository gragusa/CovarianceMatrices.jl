## Tests for the CovarianceMatrix result type.

using Test
using CovarianceMatrices
using CovarianceMatrices: bandwidth, estimator, information, kernelweights
using LinearAlgebra
using DataFrames
using GLM
using PDMats
using Random
using StableRNGs

@testset "CovarianceMatrix result type" begin
    X = randn(StableRNG(11), 200, 3)

    @testset "behaves as an AbstractMatrix" begin
        V = aVar(Bartlett{Andrews}(), X)
        @test V isa AbstractMatrix
        @test size(V) == (3, 3)
        @test eltype(V) == Float64
        @test V[1, 2] == V[2, 1]
        @test axes(V) == (Base.OneTo(3), Base.OneTo(3))

        M = parent(V)
        @test M isa Matrix{Float64}
        @test collect(V) == M

        # Operations that do not preserve the covariance interpretation degrade
        # to a plain matrix.
        @test V * V isa Matrix
        @test V + V isa Matrix
        @test 2V isa Matrix
        @test inv(V) isa Matrix

        b = randn(StableRNG(12), 3)
        @test V * b ≈ M * b
        @test V \ b ≈ M \ b
        @test diag(V) == diag(M)
        @test cholesky(Symmetric(parent(V))) isa Cholesky
    end

    @testset "the stored matrix is exactly symmetric" begin
        V = aVar(Bartlett{Andrews}(), X)
        @test parent(V) == transpose(parent(V))
    end

    @testset "carries the estimator and what it selected" begin
        k = Bartlett{Andrews}()
        V = aVar(k, X)
        @test estimator(V) == k
        @test bandwidth(V) > 0
        @test bandwidth(V) == information(V).bandwidth
        @test kernelweights(V) == [1.0, 1.0, 1.0]

        # A fixed bandwidth is reported as given.
        @test bandwidth(aVar(Bartlett(4), X)) == 4.0

        # Estimators that select nothing report nothing.
        Vhr = aVar(HC0(), X)
        @test bandwidth(Vhr) === nothing
        @test kernelweights(Vhr) === nothing
        @test information(Vhr) == NamedTuple()
    end

    @testset "results record the bandwidth that produced them" begin
        k = Bartlett{Andrews}()
        Y = randn(StableRNG(13), 300, 3)

        V1 = aVar(k, X)
        V2 = aVar(k, Y)
        @test bandwidth(V1) != bandwidth(V2)

        # The earlier result is untouched by the later estimate.
        @test bandwidth(V1) == bandwidth(aVar(k, X))
        @test V1 == aVar(k, X)
    end

    @testset "a reused estimator is not mutated" begin
        k = Bartlett{Andrews}()
        before = deepcopy(k)
        aVar(k, X)
        aVar(k, randn(StableRNG(14), 150, 3))
        @test k == before
        @test hash(k) == hash(before)
    end

    @testset "vcov returns a CovarianceMatrix" begin
        rng = StableRNG(15)
        df = DataFrame(X1 = randn(rng, 120), X2 = randn(rng, 120), Y = randn(rng, 120))
        m = lm(@formula(Y~X1 + X2), df)

        k = Bartlett{Andrews}()
        V = vcov(k, m)
        @test V isa CovarianceMatrix
        @test size(V) == (3, 3)
        @test estimator(V) == k
        # The bandwidth reaches the sandwich result from the inner estimate.
        @test bandwidth(V) == bandwidth(aVar(k, m))
        @test parent(V) == transpose(parent(V))

        # stderror reads through diag and is unaffected by the wrapper.
        se = stderror(k, m)
        @test se isa Vector
        @test se ≈ sqrt.(diag(V))
    end

    @testset "PDMats conversion is opt-in" begin
        V = aVar(Bartlett{Andrews}(), X)
        @test isposdef(V)
        P = PDMat(V)
        @test P isa PDMat
        b = randn(StableRNG(16), 3)
        @test invquad(P, b) ≈ dot(b, Symmetric(parent(V)) \ b)
        @test logdet(P) ≈ logdet(Symmetric(parent(V)))

        # An estimate that is not positive definite refuses to convert rather
        # than returning something that only looks like a covariance matrix.
        indefinite = CovarianceMatrices.CovarianceMatrix([1.0 2.0; 2.0 1.0], HC0())
        @test !isposdef(indefinite)
        @test_throws PosDefException PDMat(indefinite)
    end
end
