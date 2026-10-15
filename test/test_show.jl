using Test
using CovarianceMatrices
using GLM
using DataFrames

# Printed forms are parsed in a module that loads only CovarianceMatrices, so names
# such as `Truncated` resolve as a user of this package alone would see them.
module ShowRoundTrip
using CovarianceMatrices
end

roundtrip(x) = Core.eval(ShowRoundTrip, Meta.parse(repr(x)))

@testset "show" begin
    @testset "repr captures the whole output and round-trips" begin
        estimators = Any[
            Bartlett(5), Bartlett{Andrews}(), Parzen(NeweyWest), CovarianceMatrices.Truncated(3.0),
            QuadraticSpectral{Andrews}(), TukeyHanning(2), Parzen{Andrews}(),
            HC0(), HC1(), HR3(), HC4m(), HC5(), Uncorrelated(), EWC(4),
            CR0([1, 1, 2, 2]), CR1(([1, 2, 1, 2], [1, 1, 2, 2])), CR2(["a", "b", "a"]),
            CR3([1, 2, 3]), CR0(:firm), CR2((:a, :b)),
            Cluster([1, 1, 2]), Cluster((:a, :b)), Cluster(([1, 2], [3, 3])),
            VARHAC(), VARHAC(FixedLags(3)), VARHAC(BICSelector(), DifferentOwnLags([1, 2])),
            VARHAC(AICSelector(), AutoLags()),
            DriscollKraay(Bartlett(3), tis = [1, 2, 1, 2], iis = [1, 1, 2, 2]),
            DriscollKraay(Parzen{Andrews}(), tis = [1, 2, 1, 2], iis = [1, 1, 2, 2]),
            UniformSmoother(3), TriangularSmoother(0)
        ]
        for x in estimators
            r = repr(x)
            @test !isempty(r)
            @test !endswith(r, "\n")
            @test roundtrip(x) == x
        end
        @test repr(Bartlett(5)) == "Bartlett(5.0)"
        @test repr(Bartlett{Andrews}()) == "Bartlett{Andrews}()"
        @test repr(CR0([1, 1, 2, 2])) == "CR0([1, 1, 2, 2])"
        @test repr(VARHAC(AICSelector(), SameLags(4))) ==
              "VARHAC(AICSelector(), SameLags(4))"
        @test repr(VARHAC(FixedLags(3))) == "VARHAC(FixedLags(3))"
        @test "k = $(EWC(4))" == "k = EWC(4)"
    end

    @testset "cached wrappers" begin
        k = CachedCR(CR1([1, 1, 2, 2]), 3)
        @test repr(k) == "CachedCR(CR1([1, 1, 2, 2]), 3, Float64)"
        k2 = roundtrip(k)
        @test k2 isa CachedCR
        @test k2 == k
        @test k2.cache.ncols == 3

        df = DataFrame(y = [1.0, 2.0, 2.5, 4.0, 5.5, 5.0], x = 1:6, g = [1, 1, 2, 2, 3, 3])
        m = lm(@formula(y ~ x), df)
        km = CachedCR(CR0(df.g), m)
        @test repr(km) ==
              "CachedCR(CR0([1, 1, 2, 2, 3, 3]), <model cache: nobs = 6, ncols = 2>)"
    end

    @testset "CovarianceMatrix header" begin
        X = [1.0 2.0; 3.0 5.0; 2.0 1.0; 4.0 4.0; 0.5 3.0; 2.5 1.5]
        V = aVar(Bartlett(2), X)
        s = sprint(show, MIME"text/plain"(), V)
        @test startswith(s, "2×2 CovarianceMatrix from Bartlett(2.0):\n")
        # A long cluster vector is truncated in the header.
        Vc = aVar(CR0(repeat(1:500; inner = 2)), randn(1000, 2))
        header = first(split(sprint(show, MIME"text/plain"(), Vc), '\n'))
        @test occursin("…", header)
        @test length(header) < 200
    end
end
