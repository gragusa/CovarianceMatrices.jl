## A `RegressionModel` that implements only the documented protocol gets the same
## variances as a GLM fit of the same regression.

using Test
using CovarianceMatrices
using LinearAlgebra
using DataFrames
using GLM
using StableRNGs
using StatsAPI
using StatsBase: StatsBase

const CMRP = CovarianceMatrices

struct ProtocolOLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
end
ProtocolOLS(X, y) = ProtocolOLS(X, y, X \ y)
StatsAPI.coef(m::ProtocolOLS) = m.β
StatsAPI.modelmatrix(m::ProtocolOLS) = m.X
StatsAPI.residuals(m::ProtocolOLS) = m.y - m.X * m.β
StatsAPI.leverage(m::ProtocolOLS) = diag(m.X * inv(m.X' * m.X) * m.X')
StatsAPI.weights(m::ProtocolOLS) = Float64[]
CMRP.bread(m::ProtocolOLS) = inv(m.X' * m.X)

# Implements the required methods and nothing optional.
struct MinimalOLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
end
MinimalOLS(X, y) = MinimalOLS(X, y, X \ y)
StatsAPI.coef(m::MinimalOLS) = m.β
StatsAPI.modelmatrix(m::MinimalOLS) = m.X
CMRP.momentresiduals(m::MinimalOLS) = m.y - m.X * m.β
CMRP.bread(m::MinimalOLS) = inv(m.X' * m.X)

# Returns a leverage vector it stores, which the adjustments must not modify.
struct StoredLeverageOLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    h::Vector{Float64}
end
StoredLeverageOLS(X, y) = StoredLeverageOLS(X, y, X \ y, diag(X * inv(X' * X) * X'))
StatsAPI.coef(m::StoredLeverageOLS) = m.β
StatsAPI.modelmatrix(m::StoredLeverageOLS) = m.X
StatsAPI.residuals(m::StoredLeverageOLS) = m.y - m.X * m.β
StatsAPI.leverage(m::StoredLeverageOLS) = m.h
CMRP.bread(m::StoredLeverageOLS) = inv(m.X' * m.X)

# Defines `numobs` instead of relying on the number of rows of `modelmatrix`.
struct CountedOLS <: StatsBase.RegressionModel
    inner::ProtocolOLS
    n::Int
end
StatsAPI.coef(m::CountedOLS) = coef(m.inner)
StatsAPI.modelmatrix(m::CountedOLS) = modelmatrix(m.inner)
StatsAPI.residuals(m::CountedOLS) = residuals(m.inner)
CMRP.bread(m::CountedOLS) = CMRP.bread(m.inner)
CMRP.numobs(m::CountedOLS) = m.n

struct ProtocolMLE <: MLikeModel
    Z::Matrix{Float64}
    H::Matrix{Float64}
end
StatsAPI.coef(m::ProtocolMLE) = zeros(size(m.Z, 2))
StatsAPI.nobs(m::ProtocolMLE) = size(m.Z, 1)
CMRP.momentmatrix(m::ProtocolMLE) = m.Z
CMRP.hessian_objective(m::ProtocolMLE) = m.H

@testset "Regression-model protocol" begin
    rng = StableRNG(20261015)
    n = 80
    df = DataFrame(x = randn(rng, n), g = repeat(1:8, inner = 10), h = repeat(1:5, 16))
    df.y = 1 .+ 0.5 .* df.x .+ randn(rng, n)
    glmfit = lm(@formula(y ~ x), df)
    X = modelmatrix(glmfit)
    ols = ProtocolOLS(X, df.y)

    @testset "protocol names are public" begin
        for s in (:bread, :momentresiduals, :leverage, :mask)
            @test Base.ispublic(CovarianceMatrices, s)
            @test !Base.isexported(CovarianceMatrices, s)
        end
    end

    @testset "fallbacks" begin
        @test CMRP.momentresiduals(ols) ≈ residuals(glmfit)
        @test CMRP.leverage(ols) ≈ CMRP.leverage(glmfit)
        @test CMRP.mask(ols) == [true, true]
    end

    @testset "matches GLM: $(nameof(typeof(k)))" for k in (
        HC0(), HC1(), HC2(), HC3(), HC4(), HC4m(), HC5(), Bartlett(3),
        Bartlett{Andrews}(), Parzen{NeweyWest}(), EWC(4), VARHAC())
        V = vcov(k, ols)
        @test V isa CovarianceMatrix
        @test V ≈ vcov(k, glmfit)
        @test stderror(k, ols) ≈ stderror(k, glmfit)
    end

    @testset "matches GLM: $(nameof(typeof(k)))" for k in (
        CR0(df.g), CR1(df.g), CR2(df.g), CR3(df.g), CR1((df.g, df.h)))
        @test vcov(k, ols) ≈ vcov(k, glmfit)
    end

    @testset "model with only the required methods" begin
        minimal = MinimalOLS(X, df.y)
        @test vcov(HC1(), minimal) ≈ vcov(HC1(), glmfit)
        @test vcov(CR1(df.g), minimal) ≈ vcov(CR1(df.g), glmfit)
        # The leverage-based estimators need `leverage`.
        @test_throws MethodError vcov(HC3(), minimal)
        @test !isdefined(CovarianceMatrices, :_residuals)
    end

    @testset "GLM fits use the weighted working residuals" begin
        w = 0.5 .+ rand(rng, n)
        wfit = lm(@formula(y ~ x), df; wts = w)
        @test CMRP.momentresiduals(wfit) ≈ w .* residuals(wfit)
        @test CMRP.numobs(wfit) == n
    end

    @testset "vcov(Information(), model) for an MLikeModel" begin
        Z = randn(rng, n, 2)
        mle = ProtocolMLE(Z, Z' * Z)
        V = vcov(Information(), mle)
        @test V isa CovarianceMatrix
        @test estimator(V) == Information()
        @test V ≈ inv(Z' * Z)
        @test stderror(Information(), mle) ≈ sqrt.(diag(V))
        @test_throws MethodError vcov(HC0(), Information(), mle)
    end
end

## A package that forms its own moment matrices (as Regress.jl does) reproduces
## `vcov(k, model)` with the public extension surface.
@testset "Extension surface for modeling packages" begin
    rng = StableRNG(20261016)
    n = 80
    df = DataFrame(x = randn(rng, n), g = repeat(1:8, inner = 10),
        unit = repeat(1:5, 16), period = repeat(1:16, inner = 5))
    df.y = 1 .+ 0.5 .* df.x .+ randn(rng, n)
    glmfit = lm(@formula(y ~ x), df)
    X = modelmatrix(glmfit)
    ols = ProtocolOLS(X, df.y)
    u = CMRP.momentresiduals(ols)
    B = CMRP.bread(ols)

    @testset "names are public and not exported" begin
        for s in (:residual_adjustment, :numobs, :bandwidthweights, :clusters, :groups,
            :cache, :timekernel, :timeindex, :unitindex, :correction)
            @test Base.ispublic(CovarianceMatrices, s)
            @test !Base.isexported(CovarianceMatrices, s)
        end
    end

    @testset "residual_adjustment: $(nameof(typeof(k)))" for k in (
        HC0(), HC1(), HC2(), HC3(), HC4(), HC4m(), HC5())
        a = CMRP.residual_adjustment(k, ols)
        M = X .* (a .* u)
        @test n .* B * aVar(HC0(), M; demean = false) * B ≈ vcov(k, glmfit)
    end

    @testset "residual_adjustment values" begin
        h = CMRP.leverage(ols)
        @test CMRP.residual_adjustment(HC0(), ols) == 1
        @test CMRP.residual_adjustment(HC1(), ols) ≈ sqrt(n / (n - 2))
        @test CMRP.residual_adjustment(HC2(), ols) ≈ 1 ./ sqrt.(1 .- h)
        @test CMRP.residual_adjustment(HC3(), ols) ≈ 1 ./ (1 .- h)
        @test CMRP.residual_adjustment(Bartlett(3), ols) == 1
    end

    @testset "a stored leverage vector is not modified" begin
        stored = StoredLeverageOLS(X, df.y)
        h0 = copy(stored.h)
        for k in (HC4(), HC4m(), HC5())
            V1 = vcov(k, stored)
            V2 = vcov(k, stored)
            @test stored.h == h0
            @test V1 == V2
            @test V1 ≈ vcov(k, glmfit)
        end
    end

    @testset "numobs" begin
        @test CMRP.numobs(ols) == n
        counted = CountedOLS(ols, 40)
        @test CMRP.residual_adjustment(HC1(), counted) ≈ sqrt(40 / 38)
    end

    @testset "bandwidthweights: $(nameof(typeof(k)))" for k in (
        Bartlett{Andrews}(), Parzen{NeweyWest}(), QuadraticSpectral{Andrews}())
        kw = CMRP.bandwidthweights(k, X)
        @test kw == [0.0, 1.0]
        M = X .* u
        A = aVar(k, M; demean = false, weights = kw)
        Am = aVar(k, ols)
        @test A ≈ Am
        @test bandwidth(A) == bandwidth(Am)
        @test kernelweights(A) == kw
        # Weights computed from the moment matrix count the intercept column.
        @test kernelweights(aVar(k, M; demean = false)) == [1.0, 1.0]
    end

    @testset "bandwidthweights is nothing without data-driven bandwidth" begin
        for k in (Bartlett(3), HC1(), CR0(df.g), EWC(4), VARHAC())
            @test CMRP.bandwidthweights(k, X) === nothing
        end
    end

    @testset "DriscollKraay accessors" begin
        kernel = Bartlett{Andrews}()
        dk = DriscollKraay(kernel, df.period, df.unit; correction = :HC1)
        @test CMRP.timekernel(dk) == kernel
        @test CMRP.groups(CMRP.timeindex(dk)) == df.period
        @test CMRP.nclusters(CMRP.timeindex(dk)) == 16
        @test CMRP.groups(CMRP.unitindex(dk)) == df.unit
        @test CMRP.correction(dk) === :HC1
        @test CMRP.correction(DriscollKraay(kernel, df.period, df.unit)) === :HC0
        @test CMRP.correction(DriscollKraay(kernel; tis = :t, iis = :id,
            correction = :sss)) === :sss
        # Rebuilt without the correction, which `aVar` on a matrix requires.
        dk0 = DriscollKraay(CMRP.timekernel(dk), CMRP.timeindex(dk), CMRP.unitindex(dk))
        @test dk0.correction === :HC0
        A = aVar(dk0, X .* u; demean = false, weights = CMRP.bandwidthweights(dk, X))
        Am = aVar(dk, ols)
        @test A ≈ Am
        @test bandwidth(A) == bandwidth(Am)
        @test_throws ArgumentError aVar(dk, X .* u)
    end

    @testset "cluster accessors" begin
        k = CR1((df.g, df.unit))
        cs = CMRP.clusters(k)
        @test length(cs) == 2
        @test all(c -> c isa CMRP.Clustering, cs)
        @test CMRP.groups(cs[1]) == df.g
        @test map(CMRP.nclusters, cs) == CMRP.nclusters(k) == (8, 5)
        @test CMRP.clusters(CR1(:firm, :year)) == (:firm, :year)
        @test CMRP.clusters(Cluster(:firm)) == (:firm,)
        # The matrix-path estimator on the same clusters gives the CR0 variance.
        M = X .* u
        @test aVar(CR0(cs), M) ≈ aVar(CR0((df.g, df.unit)), M)
    end

    @testset "CachedCR accessors" begin
        k = CR1(df.g)
        ck = CachedCR(k, size(X, 2))
        @test estimator(ck) == k
        @test CMRP.clusters(ck) === CMRP.clusters(k)
        # The cache of a matrix-path `CachedCR` serves CR0 on the same clusters.
        c0 = CachedCR(CR0(CMRP.clusters(ck)), CMRP.cache(ck))
        M = X .* u
        @test aVar(c0, M) ≈ aVar(CR0(df.g), M)
        @test CMRP.cache(c0) === CMRP.cache(ck)
    end
end
