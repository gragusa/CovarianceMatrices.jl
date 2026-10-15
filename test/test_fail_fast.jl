## Invalid inputs throw errors that name the problem, and numerical fallbacks warn.

using Test
using CovarianceMatrices
using LinearAlgebra
using DataFrames
using GLM
using StableRNGs
using StatsAPI

struct FailFastThrowingCoef end
StatsAPI.coef(::FailFastThrowingCoef) = throw(DomainError(-1, "coef failed"))

@testset "Fail fast" begin
    rng = StableRNG(20261016)
    n = 60
    X = randn(rng, n, 2)
    df = DataFrame(y = randn(rng, n), x = randn(rng, n), g = repeat(1:6, inner = 10))
    model = lm(@formula(y ~ x), df)

    @testset "Cluster lengths" begin
        short = repeat(1:5, inner = 10)
        @test_throws "the cluster variables have 50 observations but the moment matrix has 60 rows" aVar(
            CR0(short), X)
        @test_throws DimensionMismatch aVar(Cluster(short), X)
        @test_throws "all cluster variables must have the same length" aVar(
            CR0((df.g, short)), X)
        @test_throws DimensionMismatch aVar(CR0((df.g, df.g, short)), X)
        @test_throws "clustering needs at least 2 observations" aVar(CR0([1]), ones(1, 2))
        @test_throws "the cluster variable has 50 observations but the moment matrix has 60 rows" vcov(
            CR0(short), model)
        @test_throws DimensionMismatch vcov(CR1((df.g, short)), model)

        cached = CovarianceMatrices.CachedCR(CR0(df.g), 2)
        @test_throws "the moment matrix has 3 columns but the cache was built for 2" aVar(
            cached, randn(rng, n, 3))
        @test_throws DimensionMismatch aVar(cached, randn(rng, n - 1, 2))
    end

    @testset "Fixed bandwidths" begin
        for K in (Bartlett, Parzen, CovarianceMatrices.Truncated, TukeyHanning, QuadraticSpectral)
            @test_throws ArgumentError K(-1)
            @test_throws ArgumentError K(NaN)
            @test_throws ArgumentError K(Inf)
        end
        @test_throws "must be finite and positive, got 0.0" QuadraticSpectral(0.0)
        @test_throws "must be finite and nonnegative, got -1" Bartlett(-1)
        # Zero lags: the estimate is the outer product of the moments.
        for K in (Bartlett, Parzen, CovarianceMatrices.Truncated, TukeyHanning)
            @test aVar(K(0), X) ≈ aVar(HC0(), X)
        end
        # Data-driven kernels store an unused bandwidth.
        @test Bartlett{Andrews}() isa Bartlett
        @test QuadraticSpectral{NeweyWest}() isa QuadraticSpectral
        @test_throws "needs a bandwidth, as in `Bartlett(5.0)`" Bartlett{Fixed}()
    end

    @testset "Fixed bandwidth values" begin
        for K in (Bartlett, Parzen, CovarianceMatrices.Truncated, TukeyHanning, QuadraticSpectral)
            @test K(Fixed(5.0)) == K(5.0)
            @test K(Fixed(5)) isa HAC{Fixed}
            @test bandwidth(K(Fixed(3))) == 3.0
        end
        @test bandwidth(Fixed(4)) == 4.0
        @test eval(Meta.parse(repr(Fixed(2.5)))) == Fixed(2.5)
        @test_throws "a fixed bandwidth must be finite and nonnegative, got -1" Fixed(-1)
        @test_throws ArgumentError Fixed(NaN)
        @test_throws ArgumentError Fixed(Inf)
        @test_throws "must be finite and positive, got 0.0" QuadraticSpectral(Fixed(0))
        @test aVar(Bartlett(Fixed(3)), X) == aVar(Bartlett(3), X)
    end

    @testset "Newey-West rule on Truncated and TukeyHanning" begin
        msg = "the Newey-West (1994) bandwidth rule is defined only for the Bartlett, Parzen and QuadraticSpectral kernels"
        for K in (CovarianceMatrices.Truncated, TukeyHanning)
            @test_throws msg K{NeweyWest}()
            @test_throws msg K(NeweyWest)
            @test_throws msg K{NeweyWest}(1.0)
        end
    end

    @testset "Bandwidth given to a data-driven kernel" begin
        @test (@test_deprecated r"will throw" Bartlett{Andrews}(5.0)) == Bartlett{Andrews}()
        @test (@test_deprecated Parzen{NeweyWest}(2)) == Parzen{NeweyWest}()
    end

    @testset "kernelweights and optimalbw signatures" begin
        V = aVar(Bartlett{Andrews}(), [ones(n) randn(rng, n, 2)])
        @test kernelweights(V) == [0.0, 1.0, 1.0]
        @test (@test_deprecated kernelweights(Bartlett{Andrews}(), X)) == ones(size(X, 2))
        @test (@test_deprecated kernelweights(HC0(), X)) === nothing
        @test (@test_deprecated optimalbw(UniformSmoother(0), 1000)) ≈ 2.0 * 1000^(1 / 3)
        @test (@test_deprecated optimalbw(TriangularSmoother(0), 1000)) ≈ 1.5 * 1000^(1 / 5)
        @test_throws "selects its bandwidth from the data" bandwidth(Bartlett{Andrews}())
    end

    @testset "Degrees-of-freedom corrections" begin
        one_cluster = fill(1, n)
        @test_throws "needs at least 2 clusters, got 1" vcov(CR1(one_cluster), model)
        @test vcov(CR0(one_cluster), model) isa CovarianceMatrices.CovarianceMatrix

        dfk = DataFrame(y = randn(rng, 2), x = randn(rng, 2))
        exact = lm(@formula(y ~ x), dfk)
        @test_throws "more observations than coefficients, got n = 2 and k = 2" vcov(
            HC1(), exact)
        @test_throws ArgumentError vcov(Bartlett(1), exact)
    end

    @testset "Exception types" begin
        @test_throws "AutoLags requires data dimensions" CovarianceMatrices.maxlags(
            VARHAC(AICSelector(), AutoLags()))
        @test_throws ArgumentError CovarianceMatrices.maxlags(
            VARHAC(AICSelector(), AutoLags()))
        # Errors raised inside an implemented method propagate unchanged.
        @test_throws DomainError CovarianceMatrices._check_coef(FailFastThrowingCoef())
    end

    @testset "prewhite" begin
        V = @test_deprecated aVar(HC0(), X; prewhite = true)
        @test V ≈ aVar(HC0(), X)
        @test_deprecated aVar(CR0(df.g), X; prewhite = true)
        @test_deprecated vcov(CR1(df.g), model; prewhite = true)
        @test_deprecated aVar(VARHAC(), X; prewhite = true)
        @test_nowarn aVar(HC0(), X; prewhite = false)

        # Estimators that prewhiten receive the request.
        Z = randn(rng, 200, 2)
        dk = DriscollKraay(Bartlett(3), tis = repeat(1:50, inner = 4),
            iis = repeat(1:4, outer = 50))
        for k in (EWC(4), UniformSmoother(5), TriangularSmoother(5), dk)
            @test !(aVar(k, Z; prewhite = true) ≈ aVar(k, Z))
        end
    end

    @testset "cluster estimators named by column" begin
        msg = "names its clusters by column"
        for k in (CR0(:g), CR1(:g, :h), CR2(:g), CR3(:g), Cluster(:g), Cluster(:g, :h))
            @test_throws msg aVar(k, X)
            @test_throws msg vcov(k, model)
            @test_throws msg stderror(k, model)
            @test vcov(k) isa VcovSpec
        end
        @test_throws "e.g. `CR1(df.g)`" aVar(CR1(:g), X)
        @test_throws msg aVar(CR0(:g), model)
        @test_throws msg CovarianceMatrices.nclusters(CR0(:g))
        @test_throws msg CovarianceMatrices.CachedCR(CR0(:g), 2)
        @test_throws msg CovarianceMatrices.CachedCRModel(CR1(:g), model)
        @test Cluster(:g) == Cluster((:g,))
        @test Cluster(:g, :h) == Cluster((:g, :h))
    end

    @testset "String on estimator types" begin
        @test (@test_deprecated String(QuadraticSpectral)) == "Quadratic Spectral"
        @test (@test_deprecated String(HC0)) == "HR0"
        @test (@test_deprecated String(DriscollKraay)) == "Driscoll-Kraay"
    end

    @testset "VARHAC selector and lag strategy" begin
        sel = CovarianceMatrices.order
        for s in (AICSelector(), BICSelector()),
            l in (SameLags(3), DifferentOwnLags([2, 1]), AutoLags())

            k = @test_nowarn VARHAC(s, l)
            V = aVar(k, X)
            @test sel(V) == (s isa AICSelector ? CovarianceMatrices.order_aic(V) :
                   CovarianceMatrices.order_bic(V))
        end
        V = aVar(@test_nowarn(VARHAC(FixedLags(2))), X)
        @test sel(V) == [2]

        # `FixedSelector` with a searching strategy selects by BIC and reports that order.
        kf = @test_deprecated VARHAC(FixedSelector(), SameLags(3))
        Vf = aVar(kf, X)
        @test Vf ≈ aVar(VARHAC(BICSelector(), SameLags(3)), X)
        @test sel(Vf) == CovarianceMatrices.order_bic(Vf)
        @test (@test_deprecated VARHAC(:fixed)) isa VARHAC{FixedSelector, SameLags}

        @test (@test_deprecated VARHAC(FixedSelector(), FixedLags(3))) ==
              VARHAC(FixedLags(3))
        @test (@test_deprecated r"will throw" VARHAC(BICSelector(), FixedLags(3))) ==
              VARHAC(FixedLags(3))
        @test (@test_deprecated VARHAC(; T = Float32)) isa
              VARHAC{AICSelector, SameLags, Float32}

        @test (@test_deprecated FixedLags(5.7)).maxlag == 6
        @test (@test_deprecated SameLags(5.7)).maxlag == 6
        @test (@test_deprecated DifferentOwnLags((2.6, 1))).maxlags == [3, 1]
        @test (@test_nowarn FixedLags(5.0)).maxlag == 5

        @test_throws "at least 1, got -1" FixedLags(-1)
        @test_throws "at least 1, got 0" FixedLags(0)
        @test_throws "at least 1, got -2" SameLags(-2)
        @test_throws "must be non-negative" DifferentOwnLags([-1, 2])
        @test_throws "got 3 entries" DifferentOwnLags([2, 1, 1])
        @test_throws "got 1 entries" DifferentOwnLags((1,))
        @test DifferentOwnLags([0, 2]).maxlags == [0, 2]
    end

    @testset "VARHAC keywords and fallbacks" begin
        @test_deprecated r"does not use the keyword\(s\) foo" aVar(VARHAC(), X; foo = 1)
        @test_nowarn aVar(VARHAC(), X)

        # Collinear columns make the lag regressions and `I - ΣAⱼ` singular.
        Y = randn(rng, 200)
        Xc = [Y Y]
        @test_logs (:warn, r"singular cross-product matrix") match_mode=:any aVar(
            VARHAC(AICSelector(), SameLags(2)), Xc)
    end
end
