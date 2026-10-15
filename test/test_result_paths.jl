## Every `aVar` and `vcov` path returns a `CovarianceMatrix` that carries the
## quantities its estimator selected.

using Test
using CovarianceMatrices
using CovarianceMatrices: CovarianceMatrix, bandwidth, estimator, information, nclusters
using LinearAlgebra
using DataFrames
using GLM
using StableRNGs

struct ResultPathsMLE <: CovarianceMatrices.MLikeModel
    g::Matrix{Float64}
    H::Matrix{Float64}
end
CovarianceMatrices.momentmatrix(m::ResultPathsMLE) = m.g
CovarianceMatrices.hessian_objective(m::ResultPathsMLE) = m.H

struct ResultPathsGMM <: CovarianceMatrices.GMMLikeModel
    g::Matrix{Float64}
    G::Matrix{Float64}
    H::Matrix{Float64}
end
CovarianceMatrices.momentmatrix(m::ResultPathsGMM) = m.g
CovarianceMatrices.jacobian_momentfunction(m::ResultPathsGMM) = m.G
CovarianceMatrices.hessian_objective(m::ResultPathsGMM) = m.H

@testset "CovarianceMatrix from every path" begin
    rng = StableRNG(20261015)
    n = 120
    df = DataFrame(x1 = randn(rng, n), x2 = randn(rng, n),
        firm = repeat(1:12; inner = 10), year = repeat(1:10; outer = 12))
    df.y = 1.0 .+ 0.5 .* df.x1 .- 0.3 .* df.x2 .+ randn(rng, n)
    m = lm(@formula(y ~ x1 + x2), df)
    X = randn(rng, n, 3)

    families = Any[
        HC0(), HC1(), HC3(), Uncorrelated(), Bartlett(3), Parzen{Andrews}(),
        QuadraticSpectral{NeweyWest}(), EWC(4), VARHAC(), CR0(df.firm),
        CR1((df.firm, df.year)), CR2(df.firm), CR3(df.firm),
        UniformSmoother(2), TriangularSmoother(2),
        DriscollKraay(Bartlett(2), tis = df.year, iis = df.firm),
        DriscollKraay(Bartlett{Andrews}(), tis = df.year, iis = df.firm)
    ]

    @testset "matrix path: $(repr(k))" for k in families
        V = aVar(k, X)
        @test V isa CovarianceMatrix
        @test estimator(V) == k
    end

    @testset "model path: $(repr(k))" for k in families
        A = aVar(k, m)
        @test A isa CovarianceMatrix
        V = vcov(k, m)
        @test V isa CovarianceMatrix
        @test estimator(V) == k
    end

    @testset "cluster counts" begin
        k = CR1((df.firm, df.year))
        @test information(aVar(k, X)).nclusters == (12, 10)
        @test information(aVar(k, m)).nclusters == (12, 10)
        @test information(vcov(CR0(df.firm), m)).nclusters == (12,)
    end

    @testset "Driscoll-Kraay keeps the inner kernel's bandwidth" begin
        k = DriscollKraay(Bartlett{Andrews}(), tis = df.year, iis = df.firm)
        V = aVar(k, X)
        @test bandwidth(V) isa Real
        @test bandwidth(V) > 0
        @test bandwidth(vcov(k, m)) == bandwidth(aVar(k, m; scale = false))
        @test bandwidth(aVar(DriscollKraay(Bartlett(2), tis = df.year, iis = df.firm), X)) ==
              2
        for type in (:HC0, :HC1, :sss)
            @test vcov(k, m; type) isa CovarianceMatrix
        end
    end

    @testset "cached CR on a model" begin
        k = CR2(df.firm)
        ck = CachedCRModel(k, m)
        A = aVar(ck, m)
        @test A isa CovarianceMatrix
        @test information(A).nclusters == (12,)
        @test A ≈ aVar(k, m)
        V = vcov(ck, m)
        @test V isa CovarianceMatrix
        @test V ≈ vcov(k, m)
        @test stderror(ck, m) ≈ stderror(k, m)
    end

    @testset "M- and GMM-like models" begin
        g = randn(rng, n, 2)
        H = g' * g
        mle = ResultPathsMLE(g, H)
        Vi = vcov(Bartlett{Andrews}(), Information(), mle; check = false)
        @test Vi isa CovarianceMatrix
        @test estimator(Vi) == Information()
        Vm = vcov(Bartlett{Andrews}(), Misspecified(), mle; check = false)
        @test Vm isa CovarianceMatrix
        @test estimator(Vm) == Bartlett{Andrews}()
        @test bandwidth(Vm) isa Real

        G = [2.0 0.5; 0.3 1.5]
        gmm = ResultPathsGMM(g, G, G' * G)
        for form in (Information(), Misspecified())
            V = vcov(Parzen{Andrews}(), form, gmm; check = false)
            @test V isa CovarianceMatrix
            @test estimator(V) == Parzen{Andrews}()
            @test bandwidth(V) isa Real
            Vw = vcov(HC0(), form, gmm; W = Matrix(1.0I, 2, 2), check = false)
            @test Vw isa CovarianceMatrix
        end
    end
end
