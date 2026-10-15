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
        if k isa CovarianceMatrices._ModelOnlyAdjustment
            @test_throws ArgumentError aVar(k, X)
        else
            V = aVar(k, X)
            @test V isa CovarianceMatrix
            @test estimator(V) == k
        end
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
        @test information(aVar(CR0((df.firm, df.year)), X)).nclusters == (12, 10)
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
        for correction in (:HC0, :HC1, :sss)
            kc = DriscollKraay(Bartlett{Andrews}(), tis = df.year, iis = df.firm; correction)
            @test vcov(kc, m) isa CovarianceMatrix
        end
    end

    @testset "cached CR on a model" begin
        k = CR2(df.firm)
        ck = CachedCR(k, m)
        A = aVar(ck, m)
        @test A isa CovarianceMatrix
        @test information(A).nclusters == (12,)
        @test A ≈ aVar(k, m)
        V = vcov(ck, m)
        @test V isa CovarianceMatrix
        @test V ≈ vcov(k, m)
        @test stderror(ck, m) ≈ stderror(k, m)
    end

    @testset "one cached CR on both paths: $(nameof(K))" for K in (CR0, CR1, CR2, CR3)
        for g in ((df.firm,), (df.firm, df.year))
            k = K(g...)
            cm = CachedCR(k, m)            # cache built from the model
            cx = CachedCR(k, size(X, 2))   # cache built for moment matrices
            @test cm isa CovarianceMatrices.CR && cx isa CovarianceMatrices.CR
            for c in (cm, cx)
                if K === CR0
                    @test aVar(c, X) ≈ aVar(k, X)
                else
                    @test_throws ArgumentError aVar(c, X)
                end
                @test aVar(c, m) ≈ aVar(k, m)
                @test vcov(c, m) ≈ vcov(k, m)
                @test vcov(c, m) isa CovarianceMatrix
                @test vcov(c) isa VcovSpec
            end
        end
    end

    @testset "cached CR equality ignores the cache" begin
        k = CR2(df.firm)
        cm, cx = CachedCR(k, m), CachedCR(k, size(X, 2))
        @test cm == cx
        @test hash(cm) == hash(cx)
        @test CachedCR(k, 2) == CachedCR(k, 5)
        @test cm != k
        @test cm != CachedCR(CR1(df.firm), m)
        @test length(unique([cm, cx, CachedCR(k, 7)])) == 1
    end

    @testset "cached CR checks the model against its cache" begin
        msub = lm(@formula(y ~ x1 + x2), df[1:60, :])
        cm = CachedCR(CR2(df.firm), m)
        @test_throws "the model has 60 observations and 3 coefficients, but the cache was built for $n and 3" vcov(
            cm, msub)
    end

    @testset "CachedCRModel is a deprecated alias of CachedCR" begin
        @test Base.isdeprecated(CovarianceMatrices, :CachedCRModel)
        @test CovarianceMatrices.CachedCRModel === CachedCR
    end

    @testset "M- and GMM-like models" begin
        g = randn(rng, n, 2)
        H = g' * g
        mle = ResultPathsMLE(g, H)
        Vi = vcov(Information(), mle; check = false)
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
