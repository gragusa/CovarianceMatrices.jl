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
        # A data-driven kernel carries no bandwidth.
        @test Bartlett{Andrews}() isa Bartlett
        @test QuadraticSpectral{NeweyWest}() isa QuadraticSpectral
        @test sizeof(Bartlett{Andrews}()) == 0
        @test sizeof(QuadraticSpectral{NeweyWest}()) == 0
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
        @test_throws "`Bartlett{Andrews}` selects its bandwidth from the data and takes no bandwidth, got 5.0" Bartlett{Andrews}(5.0)
        @test_throws "or `Parzen(2)` for a fixed bandwidth" Parzen{NeweyWest}(2)
        @test_throws ArgumentError Bartlett{Andrews}(Fixed(5.0))
    end

    @testset "kernelweights and optimalbw signatures" begin
        V = aVar(Bartlett{Andrews}(), [ones(n) randn(rng, n, 2)])
        @test kernelweights(V) == [0.0, 1.0, 1.0]
        @test_throws MethodError kernelweights(Bartlett{Andrews}(), X)
        @test_throws MethodError optimalbw(UniformSmoother(0), 1000)
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

    @testset "model-only corrections on a matrix" begin
        for k in (HC1(), HC2(), HC3(), HC4(), HC4m(), HC5())
            @test_throws "use `HR0` (`HC0`) on a matrix" aVar(k, X)
        end
        for k in (CR1(df.g), CR2(df.g), CR3(df.g))
            @test_throws "use `CR0` or `Cluster` on a matrix" aVar(k, X)
        end
        @test_throws "use `CR0` or `Cluster` on a matrix" aVar(
            CachedCR(CR1(df.g), size(X, 2)), X)
        # Estimators whose result on a matrix is the intended one, and the model path.
        for k in (HC0(), Uncorrelated(), CR0(df.g), Cluster(df.g),
            CachedCR(CR0(df.g), size(X, 2)))
            @test_nowarn aVar(k, X)
        end
        @test_nowarn vcov(HC3(), model)
        @test_nowarn vcov(CR2(df.g), model)
    end

    @testset "CR2/CR3 with a cluster fixed effect" begin
        # A regressor that is nonzero only within one cluster gives that cluster's
        # hat-matrix block an eigenvalue of 1, so I - H_gg is singular.
        dfe = DataFrame(y = df.y, x = df.x, g = df.g, fe = string.(df.g))
        mfe = lm(@formula(y ~ x + fe), dfe)
        @test_throws "CR2 needs I - H_gg to be nonsingular" vcov(CR2(dfe.g), mfe)
        @test_throws "as with a cluster fixed effect" vcov(CR3(dfe.g), mfe)
        @test_throws "CR3 needs I - H_gg to be nonsingular" CachedCR(CR3(dfe.g), mfe)
    end

    @testset "export tiers" begin
        @test Base.isexported(CovarianceMatrices, :CR)
        @test Cluster <: CR
        @test !isdefined(CovarianceMatrices, :a𝕍ar)
    end

    @testset "prewhite" begin
        @test_throws "not supported by HR0" aVar(HC0(), X; prewhite = true)
        @test_throws "not supported by CR0" aVar(CR0(df.g), X; prewhite = true)
        @test_throws "not supported by CR1" vcov(CR1(df.g), model; prewhite = true)
        @test_throws "not supported by CachedCR" vcov(
            CachedCR(CR1(df.g), model), model; prewhite = true)
        @test_throws "not supported by VARHAC" aVar(VARHAC(), X; prewhite = true)
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
        @test_throws msg CachedCR(CR1(:g), model)
        @test Cluster(:g) == Cluster((:g,))
        @test Cluster(:g, :h) == Cluster((:g, :h))
    end

    @testset "String on estimator types" begin
        @test_throws MethodError String(HC0)
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

        FS = CovarianceMatrices.FixedSelector
        @test :FixedSelector ∉ names(CovarianceMatrices)
        @test_throws "`FixedSelector` is not a lag-order criterion" VARHAC(FS(), SameLags(3))
        @test_throws "Use :aic or :bic, or `VARHAC(FixedLags(k))`" VARHAC(:fixed)
        @test_throws "takes no selector. Use `VARHAC(FixedLags(3))`" VARHAC(
            FS(), FixedLags(3))
        @test_throws "takes no selector. Use `VARHAC(FixedLags(3))`" VARHAC(
            BICSelector(), FixedLags(3))
        @test_throws "takes no selector" VARHAC(AICSelector(), FixedLags(3))
        @test_throws MethodError VARHAC(; T = Float32)
        @test_throws "too many parameters" VARHAC{AICSelector, SameLags, Float32}

        @test_throws "a lag order must be an integer, got 5.7" FixedLags(5.7)
        @test_throws "a lag order must be an integer, got 5.7" SameLags(5.7)
        @test_throws "a lag order must be an integer, got 2.6" DifferentOwnLags((2.6, 1))
        @test (@test_nowarn FixedLags(5.0)).maxlag == 5
        @test (@test_nowarn DifferentOwnLags((2.0, 1))).maxlags == [2, 1]

        @test_throws "at least 1, got -1" FixedLags(-1)
        @test_throws "at least 1, got 0" FixedLags(0)
        @test_throws "at least 1, got -2" SameLags(-2)
        @test_throws "must be non-negative" DifferentOwnLags([-1, 2])
        @test_throws "got 3 entries" DifferentOwnLags([2, 1, 1])
        @test_throws "got 1 entries" DifferentOwnLags((1,))
        @test DifferentOwnLags([0, 2]).maxlags == [0, 2]
    end

    @testset "VARHAC keywords and fallbacks" begin
        @test_throws "does not accept the keyword(s) foo, bar" aVar(
            VARHAC(), X; foo = 1, bar = 2)
        @test_nowarn aVar(VARHAC(), X)

        # Collinear columns make the lag regressions and `I - ΣAⱼ` singular.
        Y = randn(rng, 200)
        Xc = [Y Y]
        @test_logs (:warn, r"singular cross-product matrix") match_mode=:any aVar(
            VARHAC(AICSelector(), SameLags(2)), Xc)
    end
end
