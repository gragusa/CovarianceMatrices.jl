using Test
using CovarianceMatrices
using LinearAlgebra
using OffsetArrays
using StableRNGs
using DataFrames
using GLM

# Every entry point accepts views and rejects offset axes. The views below have
# parents that hold other data, so reading the parent instead of the view changes
# the result.
@testset "generic axes" begin
    rng = StableRNG(2718)
    n = 60
    X = randn(rng, n, 3)
    rows = view(vcat(5 .* randn(rng, 10, 3), X, 5 .* randn(rng, 10, 3)), 11:(n + 10), :)
    cols = view(hcat(randn(rng, n, 2), X), :, 3:5)
    strided = view(repeat(X; inner = (2, 1)), 1:2:(2n), :)
    wrapped = (rows, cols, strided)
    Xo = OffsetArray(X, -2, 0)
    g = repeat(1:10; inner = 6)
    tis = repeat(1:6, 10)

    estimators = (Bartlett(4), Bartlett{Andrews}(), Parzen{Andrews}(),
        Bartlett{NeweyWest}(), Parzen{NeweyWest}(), QuadraticSpectral(4.0),
        QuadraticSpectral{Andrews}(), CovarianceMatrices.Truncated(3), TukeyHanning(3),
        EWC(5), VARHAC(), HR0(), CR0(g), CR0((g, tis)),
        DriscollKraay(Bartlett(2); tis, iis = g), UniformSmoother(2), TriangularSmoother(2))

    @testset "aVar: $(nameof(typeof(k)))" for k in estimators
        for demean in (true, false)
            ref = aVar(k, X; demean)
            for A in wrapped
                @test aVar(k, A; demean) ≈ ref
            end
        end
        if CovarianceMatrices._prewhitens(k)
            ref = aVar(k, X; prewhite = true)
            for A in wrapped
                @test aVar(k, A; prewhite = true) ≈ ref
            end
        end
        @test_throws "offset arrays are not supported" aVar(k, Xo)
    end

    @testset "optimalbw" begin
        for k in (Bartlett{Andrews}(), Parzen{NeweyWest}(), UniformSmoother(0)),
            prewhite in (false, true)

            k isa CovarianceMatrices.MomentSmoother && prewhite && continue
            kw = k isa CovarianceMatrices.MomentSmoother ? (;) : (; prewhite)
            ref = optimalbw(k, X; kw...)
            for A in wrapped
                @test optimalbw(k, A; kw...) ≈ ref
            end
        end
        for k in (Bartlett{Andrews}(), Parzen{NeweyWest}())
            @test_throws "offset arrays are not supported" optimalbw(k, Xo)
        end
    end

    @testset "cluster and panel identifiers" begin
        gv = view(vcat(0, g), 2:(n + 1))
        tv = view(vcat(tis, tis), (n + 1):(2n))
        @test aVar(CR0(gv), X) ≈ aVar(CR0(g), X)
        @test aVar(CR0((gv, tv)), X) ≈ aVar(CR0((g, tis)), X)
        @test aVar(DriscollKraay(Bartlett(2); tis = tv, iis = gv), X) ≈
              aVar(DriscollKraay(Bartlett(2); tis, iis = g), X)

        go = OffsetArray(g, -3)
        to = OffsetArray(tis, -3)
        @test_throws "offset arrays are not supported" CR0(go)
        @test_throws "offset arrays are not supported" CR0((g, to))
        @test_throws "offset arrays are not supported" DriscollKraay(
            Bartlett(2); tis = to, iis = g)
        @test_throws "offset arrays are not supported" DriscollKraay(
            Bartlett(2); tis, iis = go)
    end

    @testset "model fitted on a DataFrame view" begin
        data = DataFrame(y = randn(rng, 80), x1 = randn(rng, 80), x2 = randn(rng, 80))
        m = lm(@formula(y ~ x1 + x2), data[11:70, :])
        mv = lm(@formula(y ~ x1 + x2), view(data, 11:70, :))
        for k in (HC1(), Bartlett(3), Bartlett{Andrews}(), VARHAC(), CR1(g),
            UniformSmoother(2))
            @test vcov(k, mv) ≈ vcov(k, m)
        end
    end
end
